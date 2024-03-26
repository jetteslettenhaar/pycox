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

# Step 2: Get data ready (turn into tensors)
# Step 2a: Turn data into numbers --> my data is already in numbers since I used One Hot Encoding
# Step 2b: Load the data
'''
X: covariates of the model
e: whether the event (death or RFS) occurs? (1: occurs; 0: censored)
t/y: the time of event e.
'''
filepath = 'my_models/support_train_test.h5'

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

def _normalize(df, cols):
    ''' Performs min-max normalization on specified columns of DataFrame df.

    :param df: (DataFrame) the DataFrame to be normalized
    :param cols: (list) list of column names to be normalized
    '''
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Columns to normalize
norm_cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

# Normalize train and test data
_normalize(train_df, norm_cols)
_normalize(test_df, norm_cols)

print(train_df)
print(test_df)


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

# Create custom datasets
train_dataset = SurvivalDataset(train_df)
test_dataset = SurvivalDataset(test_df)

print(train_dataset)
print(test_dataset)

# Create custom dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

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
    def __init__(self, l2_reg):
        super().__init__()
        self.best_c_index = 0.0
        # We use the FullyConnectedNet as a model to replicate DeepSurv
        self.drop = 0.255
        self.dims = [14, 44, 1]
        self.model = self._build_network()
        self.model.to(device)       # Move model to GPU
        self.l2_reg = l2_reg        # Is this necessary? Or is it sufficient to only define it as the parameter above
        self.regularization = Regularization(order=2, weight_decay=self.l2_reg)
        self.mlflow_logger = MLFlowLogger(experiment_name="test_model_pytorch")
        mlflow.start_run()
        # We want to log everything (using MLflow)
        self.mlflow_logger.log_hyperparams({
            'l2_reg': l2_reg
        })

    def _build_network(self):
        ''' Builds the network according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            # adds batch normalization
            layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds SeLU activation function
            layers.append(nn.SELU())
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.047)        # Learning rate is hyperparameter!

        def lr_lambda(epoch):
            lr_decay_rate = 0.002573                                         # Learning rate decay is a hyperparameter!
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
        return concordance_index(y, risk_pred, e) # Risk_pred should have a negative sign --> done in training/validation step

    def training_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)
        self.mlflow_logger.log_metrics({'train_c_index': c_index})
        self.mlflow_logger.log_metrics({'train_loss': loss.item()})

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e)
        c_index = self.c_index(-risk_pred, y, e)

        self.mlflow_logger.log_metrics({'val_c_index': c_index})
        self.mlflow_logger.log_metrics({'val_loss': loss.item()})

        return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}

    def validation_epoch_end(self, outputs):
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        average_c_index = torch.stack([torch.tensor(x['c_index']) for x in outputs]).mean()
        val_c_index = average_c_index.item()

        # Log C-index and loss for the entire validation set
        self.mlflow_logger.log_metrics({'val_c_index_epoch': val_c_index})
        self.mlflow_logger.log_metrics({'val_loss_epoch': average_loss.item()})

        # Check if the current c-index is better than the previous best
        if average_c_index > self.best_c_index:
            self.best_c_index = average_c_index
            self.best_model_state = self.model.state_dict().copy()  # Save the model state

        print(f'Validation Epoch {self.current_epoch + 1}, Average Loss: {average_loss:.4f}')
        print(f'Epoch {self.current_epoch + 1}, Validation C-Index: {average_c_index:.4f}')
        print(self.best_c_index)

    def on_train_start(self):
        self.best_c_index = 0.0  # Initialize the best c-index to 0

    def on_train_end(self):
        print(f'Best C-Index: {self.best_c_index:.4f}')
        # Save the best model
        torch.save({'model': self.best_model_state}, 'best_model.pth')
        mlflow.end_run()

# Now lets try to actually train my model
# Define parameters  
l2_reg = 0                          # If this is not the case (l2_reg > 0), we need to make a regularisation class for the loss function to work! (see PyTorch model)
max_epochs = 500


model = Survivalmodel(l2_reg=l2_reg)
model.to(device)
trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=10, logger=model.mlflow_logger, accelerator='gpu', devices=1) # Do I need to put it on the GPU again? Or can I remove this?
trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)