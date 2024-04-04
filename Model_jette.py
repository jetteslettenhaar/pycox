# Step 1: Import all important modules

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import argparse

import torch
from torch import nn
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score

from monai.networks.nets import FullyConnectedNet

import mlflow
from pytorch_lightning.loggers import MLFlowLogger

import optuna
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------------------------------------

# Let set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set manual seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# -------------------------------------------------------------------------------------------------------------------------------------------
# Now we need to preprocess the data with this SurvivalDataset class. Several options will be available, like setting a threshold and sampling

class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, is_train, threshold_value=8760, sampling=None):
        ''' Loading data from .h5 file based on (is_train).

        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        :param threshold_value: (int) threshold value to filter out samples with 'y' values above this threshold
        :param sampling: (str) sampling strategy: "upsampling" or "downsampling"
        '''
        self.h5_file = 'my_models/simple_model_all.h5'  # Default path to .h5 file
        # loads data
        self.X, self.e, self.y = self._read_h5_file(is_train)
        # Remove NaN values
        self._drop_na()
        # normalizes data
        self._normalize()
        # apply threshold
        self._apply_threshold(threshold_value)
        # balance classes
        if is_train and sampling:
            self._balance_classes(sampling)

        print('=> load {} samples'.format(self.X.shape[0]))

    def _drop_na(self):
        ''' Drops rows with NaN values '''
        # Combine X, e, y into a DataFrame
        df = pd.DataFrame(np.concatenate([self.X, self.e, self.y], axis=1), columns=[f'x{i}' for i in range(self.X.shape[1])] + ['e', 'y'])
        # Drop rows with NaN values
        df = df.dropna()
        # Assign values back to X, e, y
        self.X = df[df.columns[:-2]].values  # Exclude 'e' and 'y' columns
        self.e = df['e'].values
        self.y = df['y'].values

    def _apply_threshold(self, threshold_value):
        ''' Filters out samples with 'y' values above the threshold value '''
        above_threshold = self.y.squeeze() > threshold_value
        self.X = self.X[~above_threshold]
        self.e = self.e[~above_threshold]
        self.y = self.y[~above_threshold]

    def _balance_classes(self, sampling):
        ''' Balances the classes using upsampling or downsampling '''
        df = pd.DataFrame(np.concatenate([self.X, self.e, self.y], axis=1), columns=['x', 'e', 'y'])
        minority = df[df['e'] == 1]
        majority = df[df['e'] == 0]
        
        if sampling == "upsampling":
            minority_upsampled = resample(minority,
                                          replace=True,
                                          n_samples=len(majority),
                                          random_state=seed)
            df = pd.concat([majority, minority_upsampled])
        elif sampling == "downsampling":
            majority_downsampled = resample(majority,
                                            replace=False,
                                            n_samples=len(minority),
                                            random_state=seed)
            df = pd.concat([minority, majority_downsampled])

        self.X = df['x'].values
        self.e = df['e'].values
        self.y = df['y'].values

    def _read_h5_file(self, is_train):
        ''' The function to parsing data from .h5 file.

        :return X: (np.array) (n, m)
            m is features dimension.
        :return e: (np.array) (n, 1)
            whether the event occurs? (1: occurs; 0: others)
        :return y: (np.array) (n, 1)
            the time of event e.
        '''
        split = 'train' if is_train else 'test'
        with h5py.File(self.h5_file, 'r') as f:
            X = f[split]['x'][()]
            e = f[split]['e'][()].reshape(-1, 1)
            y = f[split]['t'][()].reshape(-1, 1)
        return X, e, y

    def _normalize(self):
        ''' Performs normalizing X data. '''
        # Calculate mean and standard deviation along each column
        mean_values = self.X.mean(axis=0)
        std_values = self.X.std(axis=0)

        # Check for zero standard deviation and replace with epsilon
        epsilon = 1e-10  # Small value to avoid division by zero
        std_values[std_values == 0] = epsilon

        # Perform z-score normalization
        self.X = (self.X - mean_values) / std_values

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)

        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(np.array([e_item]))
        y_tensor = torch.from_numpy(np.array([y_item]))
        return {'x': X_tensor, 'y': y_tensor, 'e': e_tensor}

    def __len__(self):
        return self.X.shape[0]

# --------------------------------------------------------------------------------------------------------
# Step 3 Build a model

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class Survivalmodel(pl.LightningModule):
    def __init__(self, input_dim, dim_2, dim_3, drop, l2_reg):
        super().__init__()
        # We use the FullyConnectedNet as a model to replicate DeepSurv
        self.best_c_index = 0.0
        self.drop = drop
        self.input_dim = input_dim
        self.dim_2 = dim_2
        self.dim_3 = dim_3
        self.model = self._build_network()
        self.model.to(device)
        self.l2_reg = l2_reg
        self.regularization = Regularization(order=2, weight_decay=self.l2_reg)
        self.lr = 0.0001
        self.lr_decay_rate = 0.005

        self.mlflow_logger = MLFlowLogger(experiment_name="change_model_eval", run_name="simple_model_all")
        mlflow.start_run()
        # We want to log everything (using MLflow)
        self.mlflow_logger.log_hyperparams({
            'l2_reg': l2_reg,
            'drop': self.drop,
            'input_dim': self.input_dim,
            'dim_2': self.dim_2,
            'dim_3': self.dim_3,
            'lr': self.lr,
            'lr_decay_rate': self.lr_decay_rate
        })

    def _build_network(self):
        layers = []
        layers.append(nn.Linear(self.input_dim, self.dim_2))  # Fixed input dimension of the first linear layer
        layers.append(nn.BatchNorm1d(self.dim_2))
        layers.append(nn.SELU())
        layers.append(nn.Dropout(self.drop))
        layers.append(nn.Linear(self.dim_2, self.dim_3))  # Second linear layer with variable dimension
        layers.append(nn.BatchNorm1d(self.dim_3))
        layers.append(nn.SELU())
        layers.append(nn.Dropout(self.drop))
        layers.append(nn.Linear(self.dim_3, 1))  # Third linear layer with variable dimension
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.to(torch.float32))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)        # Learning rate is hyperparameter! (normal is 0.001)

        def lr_lambda(epoch):
            lr_decay_rate = self.lr_decay_rate                          # Learning rate decay is a hyperparameter! (normal is 0.1)
            return 1 / (1 + epoch * lr_decay_rate)                      # Inverse time decay function using epoch like in DeepSurv

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]

    def loss_fn(self, risk_pred, y, e):
        mask = torch.ones(y.shape[0], y.shape[0]).to(device)
        mask[(y.permute(*torch.arange(y.ndim - 1, -1, -1)) - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        
        # L2 regularization (not working now, we need Regularisation function!)
        if self.l2_reg > 0:
            l2_loss = self.regularization(self.model)
            return neg_log_loss + l2_loss
        else:
            return neg_log_loss
    
    def c_index(self, risk_pred, y, e):
        '''
        We want to check whether the inputs are numpy arrays (this is expected in the function concordance_index).
        If not, we have to convert them to numpy arrays. Then we can use the imported function to calculate the c-index
        NOTETHAT this is now only using uncensored data (which is not a lot, especially for test set)
        '''
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy()
        if not isinstance(risk_pred, np.ndarray):
            risk_pred = risk_pred.detach().cpu().numpy()
        if not isinstance(e, np.ndarray):
            e = e.detach().cpu().numpy()

        return concordance_index(y, risk_pred, e) # Risk_pred should have a negative sign

    def training_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'], batch['e']
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)
        self.mlflow_logger.log_metrics({'train_c_index': c_index})
        self.mlflow_logger.log_metrics({'train_loss': loss.item()})

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'], batch['e']
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)

        self.mlflow_logger.log_metrics({'val_c_index': c_index})
        self.mlflow_logger.log_metrics({'val_loss': loss.item()})
        self.log('val_c_index_objective', c_index, on_epoch=True)

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}
    
    def test_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'], batch['e']
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)

        # Log metrics for evaluation purposes
        self.log('test_c_index', c_index)
        self.log('test_loss', loss.item())
        self.mlflow_logger.log_metrics({'test_c_index': c_index})
        self.mlflow_logger.log_metrics({'test_loss': loss.item()})
        
        return {'test_loss': loss, 'test_c_index': c_index}

    def on_train_start(self):
        self.best_c_index = 0.0  # Initialize the best c-index to 0

    def on_train_end(self):
        print(f'Best C-Index: {self.best_c_index:.4f}')
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model based on clinical information"
    )
    parser.add_argument("-s", "--sampling", default="False", required=False, type=str, help="if and which sampling method to use.")

    args = parser.parse_args()
    
    # Lets create datasets and dataloaders
    # Create train dataset
    train_dataset = SurvivalDataset(is_train=True)
    test_dataset = SurvivalDataset(is_train=False)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])


    # Define the number  of folds for outer and inner cross-validation
    outer_k_folds = 5
    inner_k_folds = 3
    max_epochs = 800

    # Define outer K-fold cross-validation
    outer_kfold = KFold(n_splits=outer_k_folds, shuffle=True)

    # Outer cross-validation loop
    for fold_idx, (train_indices, test_indices) in enumerate(outer_kfold.split(combined_dataset)):
        print(f"Outer Fold: {fold_idx + 1}/{outer_k_folds}")

        # Split data into train and test for outer fold
        train_data_outer = torch.utils.data.Subset(combined_dataset, train_indices)
        test_data_outer = torch.utils.data.Subset(combined_dataset, test_indices)

        # Create custom dataloaders for outer fold
        train_dataloader_outer = DataLoader(train_data_outer, batch_size=len(train_data_outer), shuffle=True)
        test_dataloader_outer = DataLoader(test_data_outer, batch_size=len(test_data_outer))

        # Create inner K-fold cross-validation
        inner_kfold = KFold(n_splits=inner_k_folds, shuffle=True)

        for inner_fold_idx, (train_indices_inner, test_indices_inner) in enumerate(inner_kfold.split(train_data_outer)):
            print(f"\tInner Fold: {inner_fold_idx + 1}/{inner_k_folds}")

            # Split data into train and test for inner fold
            train_data_inner = torch.utils.data.Subset(train_data_outer, train_indices_inner)
            test_data_inner = torch.utils.data.Subset(train_data_outer, test_indices_inner)

            # Create custom dataloaders for inner fold
            train_dataloader_inner = DataLoader(train_data_inner, batch_size=len(train_data_inner), shuffle=True)
            test_dataloader_inner = DataLoader(test_data_inner, batch_size=len(test_data_inner))
        
            # Setup objective function for Optuna
            def objective(trial: optuna.trial.Trial):
                # Hyperparameters to be optimized
                dim_2 = trial.suggest_int("dim_2", 1, 100)
                dim_3 = trial.suggest_int("dim_3", 1, 100)
                drop = trial.suggest_float("drop", 0, 0.5)
                l2_reg = trial.suggest_float("l2_reg", 0, 20)

                # Define the actual model
                model = Survivalmodel(input_dim=int(train_dataset.X.shape[1]), dim_2=dim_2, dim_3=dim_3, drop=drop, l2_reg=l2_reg)
                model.to(device)
                trainer = pl.Trainer(max_epochs=max_epochs, logger=model.mlflow_logger, accelerator='gpu', devices=1)
                trainer.fit(model, train_dataloaders=train_dataloader_inner, val_dataloaders=test_dataloader_inner)

                # Extract validation C-index values for the final 10 epochs
                val_c_indices = []
                for epoch in range(max_epochs - 10, max_epochs):
                    # Get the validation C-index logged by LightningModule
                    val_c_index_value = trainer.callback_metrics['val_c_index_objective']
                    val_c_indices.append(val_c_index_value)

                # Calculate the average of the final 10 validation C-indices
                avg_val_c_index = np.mean(val_c_indices)

                return avg_val_c_index 

            # Create an Optuna study and optimize hyperparameters
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=3)

            # Get the best hyperparameters
            best_params = study.best_params

            # Use the best hyperparameters to train your final model for this fold
            final_model = Survivalmodel(
                input_dim=int(train_dataset.X.shape[1]),
                dim_2=best_params["dim_2"],
                dim_3=best_params["dim_3"],
                drop=best_params["drop"],
                l2_reg=best_params["l2_reg"]
            )

            trainer = pl.Trainer(max_epochs=max_epochs, logger=final_model.mlflow_logger, accelerator='gpu', devices=1)
            trainer.fit(final_model, train_dataloaders=train_dataloader_inner, val_dataloaders=test_dataloader_inner)

            # Evaluate the final model on the outer fold test set
            test_result = trainer.test(dataloaders=test_dataloader_outer)
            print(f"Test C-index for Outer Fold {fold_idx + 1}: {test_result[0]['test_c_index']}")


