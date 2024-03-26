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


# --------------------------------------------------------------------------------------------------------

# Let set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set manual seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# -------------------------------------------------------------------------------------------------------------------------------------------
# # Step 2d: Split into train and test set, but we do need to seperate the data (covariates) and corresponding labels (event, duration)
# I should also create a train and test dataloader and convert the data to a tensor to actually use it in the model 

class SurvivalDataset(Dataset):
    def __init__(self, dataframe):
        # Needed to make a dataloader in PyTorch
        self.dataframe = dataframe

    def __len__(self):
        # Needed to make a dataloader in PyTorch
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = torch.tensor(self.dataframe.iloc[idx, :-2].values, dtype=torch.float32).to(device)  # Exclude 'y' and 'e'
        y = torch.tensor(self.dataframe.iloc[idx, -2:].values, dtype=torch.float32).to(device)  # 'y' and 'e'
        return {'x': x, 'y': y}

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
    def __init__(self, in_channels, out_channels, hidden_channels, dropout, l2_reg, run_name):
        super().__init__()
        # We use the FullyConnectedNet as a model to replicate DeepSurv
        self.model = FullyConnectedNet(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, dropout=dropout)
        self.model.to(device)       # Move model to GPU
        self.l2_reg = l2_reg        # Is this necessary? Or is it sufficient to only define it as the parameter above
        self.regularization = Regularization(order=2, weight_decay=self.l2_reg)

        self.mlflow_logger = MLFlowLogger(experiment_name="test_logging_working", run_name=run_name)
        mlflow.start_run()
        # We want to log everything (using MLflow)
        self.mlflow_logger.log_hyperparams({
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_channels': hidden_channels,
            'dropout': dropout,
            'l2_reg': l2_reg
        })

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)        # Learning rate is hyperparameter!

        def lr_lambda(epoch):
            lr_decay_rate = 0.1                                         # Learning rate decay is a hyperparameter!
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
            risk_pred = risk_pred.detach().cpu().numpy().squeeze()
        if not isinstance(e, np.ndarray):
            e = e.detach().cpu().numpy()

        return concordance_index(y, -risk_pred, e) # Risk_pred should have a negative sign

    def training_epoch_end(self, outputs):

        # Calculate average loss for the entire training set
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        average_c_index = np.mean([x['c_index'] for x in outputs])

        # Log C-index for the entire training set
        self.mlflow_logger.log_metrics({'train_c_index_epoch': average_c_index})
        self.mlflow_logger.log_metrics({'train_loss_epoch': average_loss.item()})
        print(f'Training Epoch {self.current_epoch + 1}, Average Loss: {average_loss:.4f}')
        print(f'Epoch {self.current_epoch + 1}, Training C-Index: {average_c_index:.4f}')

    def validation_epoch_end(self, outputs):
        # Calculate average loss for the entire validation set
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        average_c_index = np.mean([x['c_index'] for x in outputs])

        # Log C-index for the entire validation set
        self.mlflow_logger.log_metrics({'val_c_index_epoch': average_c_index})
        self.log('val_c_index', average_c_index)                                      # This is for my objective function!!
        self.log("hp_metric", average_c_index, on_step=False, on_epoch=True)          # Log for Optuna
        self.mlflow_logger.log_metrics({'val_loss_epoch': average_loss.item()})
        print(f'Validation Epoch {self.current_epoch + 1}, Average Loss: {average_loss:.4f}')
        print(f'Epoch {self.current_epoch + 1}, Validation C-Index: {average_c_index:.4f}')
        # De output moet de lengte zijn van de hele validatie/train dataset 

        # Verplaatsen naar validation_step

    def training_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(risk_pred, y, e)

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(risk_pred, y, e)

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def on_train_end(self):
        mlflow.end_run()

max_epochs = 300
# Setup objective function for Optuna
def objective(trial: optuna.trial.Trial, run_name):
    # Hyperparameters to be optimized 
    in_channels = x_train.shape[1]
    out_channels = 1
    hidden_channels = [trial.suggest_int("hidden_size_1", 10, 100),
                       trial.suggest_int("hidden_size_2", 10, 100),
                       trial.suggest_int("hidden_size_3", 10, 100)]
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 0, 20)

    # Define the actual model
    model = Survivalmodel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, dropout=dropout, l2_reg=l2_reg, run_name="downsample")
    model.to(device)
    trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=10, logger=model.mlflow_logger, accelerator='gpu', devices=1)
    trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # Get the validation C-index logged by LightningModule
    c_index_value_val = trainer.callback_metrics['val_c_index']

    return c_index_value_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model based on clinical information"
    )
    parser.add_argument("-s", "--sampling", default="False", required=False, type=str, help="if and which sampling method to use.")

    args = parser.parse_args()
    # Step 2: Get data ready (turn into tensors)
    # Step 2a: Turn data into numbers --> my data is already in numbers since I used One Hot Encoding

    # Step 2b: Load the data
    '''
    X: covariates of the model
    e: whether the event (death or RFS) occurs? (1: occurs; 0: censored)
    t/y: the time of event e.
    '''
    filepath = 'my_models/simple_model_all.h5'

    with h5py.File(filepath, 'r') as f:
        data = {'train': {}, 'test': {}}
        for datasets in f:
            for array in f[datasets]:
                data[datasets][array] = f[datasets][array][:]

    x_train = data['train']['x']
    t_train = data['train']['t']
    e_train = data['train']['e']

    x_test = data['test']['x']
    t_test = data['test']['t']
    e_test = data['test']['e']

    columns = ['x' +str(i) for i in range(x_train.shape[1])]

    train_df = (pd.DataFrame(x_train, columns=columns)
                    .assign(y=t_train)
                    .assign(e=e_train))

    test_df = (pd.DataFrame(x_test, columns=columns)
                .assign(y=t_test)
                .assign(e=e_test))

    # Maybe put this together, if I want a validation dataset --> Possible later
    print(train_df)
    print(test_df)

    # Step 2c: Standardize covariates
    # stand_col = ['x12', 'x13', 'x14', 'x15'] # This is for model with 159 subjects
    stand_col = ['x15', 'x16', 'x17', 'x18']
    scalar = StandardScaler()
    train_df[stand_col] = scalar.fit_transform(train_df[stand_col])
    test_df[stand_col] = scalar.fit_transform(test_df[stand_col])

    print(train_df)
    print(test_df)

    # Set the threshold and remove everyone above that threshold (this cannot be true). Threshold chosen at year 2000 (24 years ago)
    # We also need to get rid of everything that is NaN
    threshold_value = 8760
    train_df = train_df[train_df['y'] <= threshold_value]
    test_df = test_df[test_df['y'] <= threshold_value]
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # -------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Now we have a very unbalanced dataset. We can make sure we have a more balanced dataset by doing upsampling of e=1, or downsampling of event=0
    We can perform upsampling by randomly duplicating instances of the minority class (e=1).
    We can perform downsampling by randomly removing instances from the majority class (e=0). 
    We are only going to resample the training set (this is apparantly good practice)
    '''

    train_minority = train_df[train_df['e'] == 1.0]
    train_majority = train_df[train_df['e'] == 0.0]

    # Upsample the minority class (e=1)
    if args.sampling == "upsampling":
        train_minority_upsampled = resample(train_minority,
                                            replace=True,
                                            n_samples=len(train_majority),
                                            random_state=seed)

        train_df = pd.concat([train_majority, train_minority_upsampled])

    if args.sampling == "downsampling":
        # Downsample the majority class (e=0)
        train_majority_downsampled = resample(train_majority,
                                            replace=False,
                                            n_samples=len(train_minority),
                                            random_state=seed)

        train_df = pd.concat([train_minority, train_majority_downsampled])


    # Create custom datasets
    train_dataset = SurvivalDataset(train_df)           # Either normal (train_df), upsampled (train_upsampled), or downsampled (train_downsampled)
    test_dataset = SurvivalDataset(test_df)

    # Create custom dataloaders
    batch_size = len(test_dataset)                     # Hyperparameter, can adjust this
    # Splitten aanpassen zodat er balans in de data blijft 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # Create an Optuna study and optimize hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, run_name=f"model_{args.sampling}" n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params

    # Use the best hyperparameters to train your final model
    # Use the best hyperparameters to train your final model
    final_model = Survivalmodel(in_channels=x_train.shape[1],           # Number of input channels
                                out_channels=1,                         # Number of output channels (1 for survival analysis)
                                hidden_channels=[best_params[f"hidden_size_{i+1}"] for i in range(3)],  # Number of output channels of each hidden layer
                                dropout=best_params["dropout"],         # Pass dropout directly
                                l2_reg=best_params["l2_reg"])           # Pass l2_reg directly

    trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=10, logger=final_model.mlflow_logger, accelerator='gpu', devices=1)
    trainer.fit(final_model,train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # # Now lets try to actually train my model
    # # Define parameters
    # in_channels = x_train.shape[1]      # Number of input channels
    # out_channels = 1                    # Number of output channels (1 for survival analysis)
    # hidden_channels = [10, 10, 10]      # Number of output channels of each hidden layer (can be adjusted)
    # dropout = 0.4                       # Hyperparameter, can be adjusted                
    # l2_reg = 2                          # If this is not the case (l2_reg > 0), we need to make a regularisation class for the loss function to work! (see PyTorch model)
    # max_epochs = 100


    # model = Survivalmodel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, dropout=dropout, l2_reg=l2_reg)
    # model.to(device)
    # trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=10, logger=model.mlflow_logger, accelerator='gpu', devices=1) # Do I need to put it on the GPU again? Or can I remove this?

    # trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

