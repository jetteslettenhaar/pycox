# Step 1: Import all important modules

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

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
# from sksurv.metrics import concordance_index_censored

from sksurv.metrics import brier_score
from sksurv.metrics import integrated_brier_score

from monai.networks.nets import FullyConnectedNet

import mlflow


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

# We need to find out where this is coming from!!! Somehow, sometimes the y is NaN
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

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

# Create custom dataloaders
batch_size = 64                     # Hyperparameter, can adjust this
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------------------------------------------------------------------------------------------

# Step 3 Build a model
# Define parameters
in_channels = x_train.shape[1]      # Number of input channels
out_channels = 1                    # Number of output channels (1 for survival analysis)
hidden_channels = [10, 10, 10]      # Number of output channels of each hidden layer (can be adjusted)
dropout = 0.1                       # Hyperparameter, can be adjusted                
l2_reg = 0                          # If this is not the case (l2_reg > 0), we need to make a regularisation class for the loss function to work! (see PyTorch model)

class Survivalmodel(pl.LightningModule):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        # We use the FullyConnectedNet as a model to replicate DeepSurv
        self.model = FullyConnectedNet(in_channels=in_channels, out_channels=1, hidden_channels=hidden_channels, dropout=dropout)
        self.model.to(device)       # Move model to GPU
        self.l2_reg = l2_reg        # Is this necessary? Or is it sufficient to only define it as the parameter above

        # We want to log everything (using MLflow)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)         # Learning rate is hyperparameter!
        return optimizer

    def loss_fn(self, risk_pred, y, e, l2_reg):
        mask = torch.ones(y.shape[0], y.shape[0]).to(device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        
        # L2 regularization (not working now, we need Regularisation function!)
        if l2_reg > 0:
            reg = Regularization(order=2, weight_decay=l2_reg)
            l2_loss = reg(self.model)
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
            threshold = 0.5 
            e = e > threshold
        return concordance_index(y, -risk_pred, e) # Risk_pred should have a negative sign

    # def c_index(self, risk_pred, y, e):
    #     if not isinstance(y, np.ndarray):
    #         y = y.detach().cpu().numpy()
    #         print(y)
    #     if not isinstance(risk_pred, np.ndarray):
    #         risk_pred = risk_pred.detach().cpu().numpy().squeeze()
    #         print(risk_pred)
    #     if not isinstance(e, np.ndarray):
    #         e = e.detach().cpu().numpy()
    #         threshold = 0.5 
    #         e = e > threshold
    #         print(e)
    #     c_harrell = concordance_index_censored(e, y, risk_pred)
    #     return c_harrell

    # def brier_score(self, risk_pred, y, e):
    #     """
    #     Calculate the brier score with the risk predictions, durations, events ad durations at which the brier score is calculated. 
    #     This function will return the Brier Score at different points in time.
    #     """
    #     # Move everything to CPU
    #     max_duration = torch.max(y.cpu()).item()  # Use torch.max() instead of np.max()
    #     time_grid = np.linspace(0, max_duration, 100)

    #     brier_scores = []

    #     for t in time_grid:
    #         mask = (y.cpu().numpy() >= t)
    #         weights = (y.cpu().numpy() >= t) & (e.cpu().numpy() == 0)
    #         risk_pred_np = risk_pred.detach().cpu().numpy()  # Use detach() instead of cpu()
    #         e_np = e.detach().cpu().numpy()  # Use detach() instead of cpu()
    #         brier_score_t = ((risk_pred_np - e_np) ** 2 * weights).sum() / len(y)
    #         brier_scores.append(brier_score_t)

    #     return pd.Series(brier_scores, index=time_grid).rename('brier_score')

    def brier_score(self, risk_pred, y, e):
        # Convert PyTorch Tensors to NumPy arrays
        y_np = y.detach().cpu().numpy()
        e_np = e.detach().cpu().numpy()
        risk_pred_np = risk_pred.detach().cpu().numpy()

        max_duration = np.max(y_np)
        time_grid = np.linspace(0, max_duration, 100)

        brier_scores = np.empty(time_grid.shape[0], dtype=float)

        for i, t in enumerate(time_grid):
            mask = (y_np >= t)
            weights = (y_np >= t) & (e_np == 0)

            brier_scores[i] = np.mean(
                np.square(risk_pred_np) * mask.astype(int) / weights.sum()
                + np.square(1.0 - risk_pred_np) * (~mask).astype(int) / (len(y) - weights.sum())
            )

        return pd.Series(brier_scores, index=time_grid, name='brier_score')
        
    # def integrated_brier_score(self, risk_pred, y, e):
    #     max_duration = torch.max(y.cpu()).item()  # Use torch.max() instead of np.max()
    #     time_grid = np.linspace(0, max_duration, 100)

    #     brier_scores = self.brier_score(risk_pred, y, e)

    #     return np.trapz(brier_scores, time_grid)

    def integrated_brier_score(self, risk_pred, y, e):
        y_np = y.detach().cpu().numpy()

        max_duration = np.max(y_np)
        time_grid = np.linspace(0, max_duration, 100)

        brier_scores = self.brier_score(risk_pred, y, e)

        return np.trapz(brier_scores, time_grid)

    def training_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e, l2_reg)
        c_index_value = self.c_index(risk_pred, y, e)
        brier_scores = self.brier_score(risk_pred, y, e)
        integrated_brier = self.integrated_brier_score(risk_pred, y, e)

        # Logging with MLflow
        mlflow.log_metric('train_loss', loss.item())
        mlflow.log_metric('train_c_index', c_index_value)
        mlflow.log_metric('train_integrated_brier', integrated_brier)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, e = batch['x'], batch['y'][:, 0], batch['y'][:, 1]
        risk_pred = self.forward(x)
        loss = self.loss_fn(risk_pred, y, e, l2_reg)
        c_index_value = self.c_index(risk_pred, y, e)
        brier_scores = self.brier_score(risk_pred, y, e)
        integrated_brier = self.integrated_brier_score(risk_pred, y, e)

        # Logging with MLflow
        mlflow.log_metric('val_loss', loss.item())
        mlflow.log_metric('val_c_index', c_index_value)
        mlflow.log_metric('val_integrated_brier', integrated_brier)

        return {'loss': loss}

    def on_train_end(self):
        mlflow.end_run()

# Now lets try to actually train my model
max_epochs = 100
model = Survivalmodel(in_channels=in_channels, out_channels=1, hidden_channels=hidden_channels, dropout=dropout)
model.to(device)
trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1) # Do I need to put it on the GPU again? Or can I remove this?

trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)