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
class SurvivalDataset(Dataset):
    ''' The dataset class performs loading data from .h5 file. '''
    def __init__(self, is_train):
        ''' Loading data from .h5 file based on (is_train).

        :param is_train: (bool) which kind of data to be loaded?
                is_train=True: loading train data
                is_train=False: loading test data
        '''
        self.h5_file = 'my_models/support_train_test.h5'  # Default path to .h5 file
        # loads data
        self.X, self.e, self.y = self._read_h5_file(is_train)
        # normalizes data
        self._normalize()

        print('=> load {} samples'.format(self.X.shape[0]))

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
        self.X = (self.X - self.X.min(axis=0)) / (self.X.max(axis=0) - self.X.min(axis=0))

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return {'x': X_tensor, 'y': y_tensor, 'e': e_tensor}

    def __len__(self):
        return self.X.shape[0]
# Create train dataset
train_dataset = SurvivalDataset(is_train=True)
test_dataset = SurvivalDataset(is_train=False)

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
        self.mlflow_logger = MLFlowLogger(experiment_name="Other_datasets", run_name="SUPPORT_REAL")
        mlflow.start_run()
        # We want to log everything (using MLflow)
        self.mlflow_logger.log_hyperparams({
            'l2_reg': l2_reg,
            'drop': self.drop,
            'dims': self.dims
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