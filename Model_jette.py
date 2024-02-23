# Step 1: Import all important modules
import torch
from torch import nn
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from monai.networks.nets import FullyConnectedNet

import mlflow
# --------------------------------------------------------------------------------------------------------

# Now writing it without functions (to test), later write this into functions as part of a class

# Step 2: Get data ready (turn into tensors)
# Step 2a: Turn data into numbers --> my data is already in numbers since I used One Hot Encoding

# Step 2b: Load the data (def load_data)
'''
X: covariates of the model
e: whether the event (death or RFS) occurs? (1: occurs; 0: censored)
t/y: the time of event e.
'''
filepath = 'my_models/simple_model.h5'
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

# Maybe put this together, if I want a validation dataset
print(train_df)
print(test_df)

# Step 2c: Standardize covariates
stand_col = ['x12', 'x13', 'x14', 'x15']
scalar = StandardScaler()
train_df[stand_col] = scalar.fit_transform(train_df[stand_col])
test_df[stand_col] = scalar.fit_transform(test_df[stand_col])
print(train_df)
print(test_df)

# # Step 2d: Split into train and test set, but we do need to seperate the data (covariates) and corresponding labels (event, duration)
# I should also create a train and test dataloader and convert the data to a tensor to actually use it in the model

class SurvivalDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = torch.tensor(self.dataframe.iloc[idx, :-2].values, dtype=torch.float32)  # Exclude 'y' and 'e'
        y = torch.tensor(self.dataframe.iloc[idx, -2:].values, dtype=torch.float32)  # 'y' and 'e'
        return {'x': x, 'y': y}

# Create custom datasets
train_dataset = SurvivalDataset(train_df)
test_dataset = SurvivalDataset(test_df)

# Create custom dataloaders
batch_size = 32  
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------------------------------------------------------------------------------------------

# Step 3 Build a model
# Define parameters
in_channels = x_train.shape[1]      # Number of input channels
out_channels = 1                    # Numer of output channels (1 for survival analysis)
hidden_channels = [10, 10, 10]      # Number of output channels of each hidden layer
dropout = 0.1                       
l2_loss = 0                         # If this is not the case, we need to make a regularisation class for the loss function to work! (see PyTorch model)

class Survivalmodel(pl.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        # We use the FullyConnectedNet as a model to replicate DeepSurv
        self.model = FullyConnectedNet(in_channels=in_channels, out_channels=1, hidden_channels=hidden_channels, dropout=dropout)

        # We want to log everything
        mlflow.start_run()
        mlflow.log_params({
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_channels': hidden_channels,
            'dropout': dropout
        })

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def loss_fn(self, risk_pred, y, e, l2reg):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        
        # L2 regularization
        if l2_reg > 0:
            reg = Regularization(order=2, weight_decay=l2_reg)
            l2_loss = reg(model)
            return neg_log_loss + l2_loss
        else:
            return neg_log_loss
    
    def c_index(self, risk_pred, y, e):
        '''
        We want to check whether the inputs are numpy arrays (this is expected in the function concordance_index).
        If not, we have to convert them to numpy arrays. Then we can use the imported function to calculate the c-index
        '''
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy()
        if not isinstance(risk_pred, np.ndarray):
            risk_pred = risk_pred.detach().cpu().numpy()
        if not isinstance(e, np.ndarray):
            e = e.detach().cpu().numpy()
        return concordance_index(y, risk_pred, e) # Maybe risk_pred should be negative sign

    def training_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e, l2reg)
        c_index_value = self.c_index(risk_pred, y, e)

        # Logging with MLflow
        mlflow.log_metric('train_loss', loss.item())
        mlflow.log_metric('train_c_index', c_index_value)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e, l2reg)
        c_index_value = self.c_index(risk_pred, y, e)

        # Logging with MLflow
        mlflow.log_metric('val_loss', loss.item())
        mlflow.log_metric('val_c_index', c_index_value)
        return {'loss': loss}

    def on_train_end(self):
        mlflow.end_run()

# Now lets try to actually train my model
max_epochs = 100
model = Survivalmodel(in_channels=in_channels, out_channels=1, hidden_channels=hidden_channels, dropout=dropout)
trainer = pl.Trainer(max_epochs=max_epochs)

trainer.fit(model,train_dataloaders=train_dataloder, val_dataloaders=val_dataloader)





