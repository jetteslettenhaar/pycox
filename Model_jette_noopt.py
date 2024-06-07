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
from lifelines.plotting import add_at_risk_counts

from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score

from monai.networks.nets import FullyConnectedNet

import mlflow
from pytorch_lightning.loggers import MLFlowLogger

import optuna
from sklearn.model_selection import KFold

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import shap


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
        self.h5_file = 'my_models/simple_model_all_AGE.h5'  # Default path to .h5 file
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
        # Set a random seed again
        torch.manual_seed(seed)
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

        self.mlflow_logger = MLFlowLogger(experiment_name="Shapley", run_name="test_simple")
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
        # self.mlflow_logger.log_metrics({'train_c_index': c_index})
        # self.mlflow_logger.log_metrics({'train_loss': loss.item()})
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_c_index', c_index, on_epoch=True)

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'], batch['e']
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)

        # self.mlflow_logger.log_metrics({'val_c_index': c_index})
        # self.mlflow_logger.log_metrics({'val_loss': loss.item()})
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_c_index', c_index, on_epoch=True)
        self.log('val_c_index_objective', c_index, on_epoch=True)

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def on_train_start(self):
        self.best_c_index = 0.0  # Initialize the best c-index to 0

    def on_train_end(self):
        print(f'Best C-Index: {self.best_c_index:.4f}')
        mlflow.end_run()

max_epochs = 800

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

    # Create custom dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # # Define the number of folds for outer cross-validation
    # outer_k_folds = 5
    # max_epochs = 800

    # # Define outer K-fold cross-validation
    # outer_kfold = KFold(n_splits=outer_k_folds, shuffle=True, random_state=seed)

    # best_hyperparams_outer_folds = []
    # test_c_indices_outer_folds = []

    # # Outer cross-validation loop
    # for fold_idx, (train_indices, test_indices) in enumerate(outer_kfold.split(combined_dataset)):
    #     print(f"Outer Fold: {fold_idx + 1}/{outer_k_folds}")

    #     # Split data into train and test for outer fold
    #     train_data_outer = torch.utils.data.Subset(combined_dataset, train_indices)
    #     test_data_outer = torch.utils.data.Subset(combined_dataset, test_indices)

    #     # Create custom dataloaders for outer fold
    #     train_dataloader_outer = DataLoader(train_data_outer, batch_size=len(train_data_outer), shuffle=True)
    #     test_dataloader_outer = DataLoader(test_data_outer, batch_size=len(test_data_outer))

    #     # Manually choose hyperparameters
    #     dim_2 = 99  # Example hyperparameter, you should choose based on prior knowledge or experimentation
    #     dim_3 = 55
    #     drop = 0.22541305037492282
    #     l2_reg = 13.152435544780317

    #     # Create and train the model with the chosen hyperparameters
    #     final_model_outer = Survivalmodel(input_dim=int(train_dataset.X.shape[1]), dim_2=dim_2, dim_3=dim_3, drop=drop, l2_reg=l2_reg)
    #     trainer_outer = pl.Trainer(max_epochs=max_epochs, logger=final_model_outer.mlflow_logger, accelerator='gpu', devices=1)
    #     trainer_outer.fit(final_model_outer, train_dataloaders=train_dataloader_outer, val_dataloaders=test_dataloader_outer)

    #     # Extract test C-index from the final training loop's metrics
    #     test_c_index_outer_fold = trainer_outer.callback_metrics['val_c_index_objective']
    #     print(f"Test C-index for Outer Fold {fold_idx + 1}: {test_c_index_outer_fold}")

    #     # Store the test C-index for this outer fold
    #     test_c_indices_outer_folds.append(test_c_index_outer_fold)

    # # Calculate the average test C-index over all outer folds
    # avg_test_c_index = np.mean(test_c_indices_outer_folds)

    # # Calculate 95% confidence interval for the test C-index
    # conf_interval = 1.96 * np.std(test_c_indices_outer_folds) / np.sqrt(len(test_c_indices_outer_folds))
    # lower_bound = avg_test_c_index - conf_interval
    # upper_bound = avg_test_c_index + conf_interval

    # print(f"Average Test C-index over {outer_k_folds} outer folds (simple): {avg_test_c_index}")
    # print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")

    '''
    Now we are going to make a KM curve based on the division in risk prediction that is made by the model. 
    '''

    # dim_2 = 95  # Example hyperparameter, you should choose based on prior knowledge or experimentation
    # dim_3 = 71
    # drop = 0.18436438636054864
    # l2_reg = 12.795963862534695

    # # Create and train the model with the chosen hyperparameters
    # final_model_KM = Survivalmodel(input_dim=int(train_dataset.X.shape[1]), dim_2=dim_2, dim_3=dim_3, drop=drop, l2_reg=l2_reg)
    # trainer_KM = pl.Trainer(max_epochs=max_epochs, logger=final_model_KM.mlflow_logger, accelerator='gpu', devices=1)
    # trainer_KM.fit(final_model_KM, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # # Obtain risk predictions from the trained model for the training dataset
    # risk_predictions_train = []
    # final_model_KM.to(device)
    # for batch in train_dataloader:
    #     x = batch['x'].to(device)
    #     risk_pred_train = final_model_KM(x)
    #     risk_predictions_train.extend(risk_pred_train.cpu().detach().numpy().flatten())
    #     print(risk_predictions_train)

    # # Obtain risk predictions from the trained model for the training dataset
    # risk_predictions_test = []
    # for batch in test_dataloader:
    #     x = batch['x'].to(device)
    #     risk_pred_test = final_model_KM(x)
    #     risk_predictions_test.extend(risk_pred_test.cpu().detach().numpy().flatten())
    #     print(risk_predictions_test)

    # # Determine the percentiles of risk predictions on the training set
    # percentile_low_train = np.percentile(risk_predictions_train, 25)
    # percentile_high_train = np.percentile(risk_predictions_train, 75)

    # # Use the percentiles obtained from the training set to categorize the risk groups on the test set
    # risk_groups_test = np.where(risk_predictions_test >= percentile_high_train, 'high-risk',
    #                             np.where(risk_predictions_test <= percentile_low_train, 'low-risk', 'unknown/intermediate'))

    # # Initialize lists to store survival data for each risk group
    # high_risk_group = []
    # low_risk_group = []

    # # Split the test dataset into high-risk and low-risk groups
    # for i, risk_pred in enumerate(risk_predictions_test):
    #     # Ensure that the loop index doesn't exceed the size of the dataset
    #     if i < len(test_dataset):
    #         if risk_groups_test[i] == 'high-risk':
    #             high_risk_group.append((test_dataset.y[i], test_dataset.e[i]))
    #         elif risk_groups_test[i] == 'low-risk':
    #             low_risk_group.append((test_dataset.y[i], test_dataset.e[i]))

    # # Create KaplanMeierFitter objects
    # kmf_high = KaplanMeierFitter()
    # kmf_low = KaplanMeierFitter()

    # # Fit the models for high-risk and low-risk groups
    # y_high, e_high = zip(*high_risk_group)
    # y_low, e_low = zip(*low_risk_group)

    # # Fit the models for high-risk and low-risk groups
    # y_high = np.array(y_high)
    # e_high = np.array(e_high)
    # y_low = np.array(y_low)
    # e_low = np.array(e_low)

    
    # # Perform log-rank test
    # results = logrank_test(y_high, y_low, e_high, e_low)

    # # Print the results
    # print("Log-rank test p-value:", results.p_value)

    # kmf_high.fit(y_high/365.25, e_high, label='High Risk Group')
    # kmf_low.fit(y_low/365.25, e_low, label='Low Risk Group')

    # # Plot the Kaplan-Meier curves
    # fig, ax = plt.subplots(figsize=(10, 6))
    # kmf_high.plot(ax=ax, color='red')
    # kmf_low.plot(ax=ax, color='green')

    # # Add at-risk counts
    # add_at_risk_counts(kmf_low, kmf_high, ax=ax)

    # # Set labels and legend
    # ax.set_xlabel('Time (years)')
    # ax.set_ylabel('Survival Probability')
    # ax.set_title('Kaplan-Meier Curve Deep Learning Model 1 (RFS)')
    # ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))

    # # Set x-axis limit to 10 years
    # ax.set_xlim(0, 10)
    # ax.set_ylim(0.4, 1)

    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('/trinity/home/r098372/pycox/figures/Thesis_figures/KM_DL_M1_RFS')

    '''
    We are going to perform a shapley analysis to see what is the most important factor
    '''
    # Assuming you have the test dataset available
    test_features = test_dataset.X
    print(test_features.shape)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)  # Convert features to PyTorch tensor

    # Step 3: Prepare the Model
    # Manually choose hyperparameters
    dim_2 = 95  # Example hyperparameter, you should choose based on prior knowledge or experimentation
    dim_3 = 71
    drop = 0.18436438636054864
    l2_reg = 12.795963862534695

    # Create and train the model with the chosen hyperparameters
    final_model_shap = Survivalmodel(input_dim=int(train_dataset.X.shape[1]), dim_2=dim_2, dim_3=dim_3, drop=drop, l2_reg=l2_reg)
    trainer_shap = pl.Trainer(max_epochs=max_epochs, logger=final_model_shap.mlflow_logger, accelerator='gpu', devices=1)
    trainer_shap.fit(final_model_shap, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    test_c_index = trainer_shap.callback_metrics['val_c_index_objective']
    print(test_c_index)

    # Step 4: Compute SHAP Values
    explainer = shap.DeepExplainer(final_model_shap, test_features_tensor)
    shap_values = explainer.shap_values(test_features_tensor, check_additivity=False)
    shap_values = shap_values.squeeze()
    print(shap_values.shape)

    # Step 5: Visualize SHAP Values
    feature_names_simple = [
        'Gender unknown', 'Female', 'Male', 'Location Colon', 'Location Duodenal', 
        'Location Esophagus', 'Location Gastric', 'Location other', 'Location Rectum', 
        'Location Small Bowel', 'Tumor status diagnosed as other', 'Tumor status diagnosed as localized', 
        'Tumor status diagnosed as advanced', 'Tumor status diagnosed as metastasized', 'Tumor status diagnosed as other 2', 
        'Age', 'Primary tumor size', 'Largest metastasis', 'Number of metastasis'
    ]

    feature_names_clinical = [
        'Gender unknown', 'Female', 'Male', 'Location Colon', 'Location Duodenal', 
        'Location Esophagus', 'Location Gastric', 'Location other', 'Location Rectum', 
        'Location Small Bowel', 'Tumor status diagnosed as other', 'Tumor status diagnosed as localized', 
        'Tumor status diagnosed as advanced', 'Tumor status diagnosed as metastasized', 'Tumor status diagnosed as other 2', 
        'Epitheloid cell', 'Mixed type', 'Spindle cell', 'Negative IHCD', 'Positive IHCD', 
        'Negative DOG', 'Positive DOG', 'KIT absent', 'KIT present', 'KIT location Exon 11', 
        'KIT location Exon 13', 'KIT location Exon 17', 'KIT location Exon 9', 'KIT No Exon location known', 'No second KIT mutation', 'Second KIT mutation', 
        'BRAF absent', 'BRAF present', 'Age', 'Primary tumor size', 'Largest metastasis', 
        'Number of metastasis', 'Mitotic count'
    ]
    
    # plt.figure()
    # shap.summary_plot(shap_values, features=test_features, feature_names=feature_names_clinical)
    # plt.savefig('/trinity/home/r098372/pycox/figures/Thesis_figures_V2/Shapley_M2_RFS.png')

    