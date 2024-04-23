'''
We will start by preparing a dictionary.
This dictionary should be one dictionary for each patient with the image_path, duration and event 0/1.
All dictionaries of all patients should be placed in a list
'''

# Lets start again by importing the important things
# import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
from torch import nn
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import nibabel as nib
from sklearn.model_selection import train_test_split

from monai.networks.nets import Densenet121
from monai.utils import set_determinism
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    ScaleIntensityRanged,
)

import matplotlib.pyplot as plt
import mlflow
from pytorch_lightning.loggers import MLFlowLogger

from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

# Import CSV file to get the labels of my images (pickle does not work with this (old) Python version)
labels = pd.read_csv('my_models/outcome_model_imaging.csv', delimiter=',')
labels = labels.drop(columns=['Censored_death', 'Censored_RFS'])
print(labels)

# Now lets put everything in one dataframe (both labels and nifti files)
labels_death = labels[['participant_id', 'Duration_death', 'Event_death']]
labels_RFS = labels[['participant_id', 'Duration_RFS', 'Event_RFS']]

# Lets make a dictionary of the subject, imagepath, duration and event!
image_path = '/data/scratch/r098372/beelden'

patient_info_list = []

for subject_name in os.listdir(image_path):
    if os.path.isdir(os.path.join(image_path, subject_name)):
        # Remove leading zero from my subject names so they match with other df
        subject_series = pd.Series([subject_name])
        subject_series_split = subject_series.str.split('_')
        subject_series = subject_series_split.str[0] + '_' + subject_series_split.str[1].str.lstrip('0')
        # Convert the subject_series to a DataFrame with the same column name as labels_death
        subject_df = pd.DataFrame({'participant_id': subject_series})
        # Use str.replace on the subject_series to remove leading zeroes
        subject_info = labels_death.merge(subject_df, on='participant_id', how='inner')
        print(subject_info)

        # Check if the 'NIFTI' directory exists for the current subject
        nifti_dir = os.path.join(image_path, subject_name, 'NIFTI')
        if os.path.exists(nifti_dir) and os.path.isdir(nifti_dir):
            # Check if there are files in the 'NIFTI' directory
            nifti_files = os.listdir(nifti_dir)
            if nifti_files:
                nifti_path = os.path.join(nifti_dir, nifti_files[0])
                print("NIFTI file path:", nifti_path)
            else:
                print("No files found in the 'NIFTI' directory for subject:", subject_name)
        else:
            print("'NIFTI' directory not found for subject:", subject_name)

    # Create the dictionary
    patient_dict = {
        'name': subject_name,
        'img': nifti_path,
        'duration': subject_info['Duration_death'].values[0],
        'event': subject_info['Event_death'].values[0]
    }

    # Append this all to a list
    patient_info_list.append(patient_dict)

# Print the list of patient dictionaries
for patient_dict in patient_info_list:
    # Make the labels floats, so the model can actually use them
    duration_str = patient_dict['duration']
    duration_str_without_days = duration_str.replace('days', '').strip()
    patient_dict['duration'] = float(duration_str_without_days)
    patient_dict['event'] = patient_dict['event'].astype(float)
    print(patient_dict)


# Let set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set manual seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# We need to determine the spacing and max image after respacing for my transforms
# Calculate median spacing in each direction
spacings = []
for data in patient_info_list:
    # Load image and get spacing
    img_path = data['img']
    img = nib.load(img_path)
    spacing = img.header.get_zooms()
    spacings.append(spacing)
median_spacing = np.median(spacings, axis=0)
print(median_spacing)

# Assuming you already have `median_spacing` calculated
# Initialize variables to store maximum dimensions
max_image_size = [0, 0, 0]                      # [max_height, max_width, max_depth]

# Iterate through each image
for data in patient_info_list:
    # Load image
    img_path = data['img']
    img = nib.load(img_path)
    
    # Rescale image using median spacing
    spacing = img.header.get_zooms()
    rescaled_shape = np.ceil(np.array(img.shape) * (np.array(spacing) / median_spacing)).astype(int)
    
    # Update maximum dimensions if necessary
    max_image_size = [max(max_image_size[i], rescaled_shape[i]) for i in range(3)]

# Print the maximum image size
print("Maximum Image Size after Rescaling:", max_image_size)

# -------------------------------------------------------------------------------------------------------------------------------
'''
The data has been prepared in a dictionary! 
Now we need to actually make the model which contains 'prepare_data' function to actually load the dictionary.
'''

# We need a regularization class just like in the other model to construct the loss function
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

class SurvivalImaging(pl.LightningModule):
    def __init__(self, l2_reg):
        super().__init__()
        self.model = Densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1
        )
        self.model.to(device)
        self.l2_reg = l2_reg
        self.regularization = Regularization(order=2, weight_decay=self.l2_reg)
        self.lr = 0.01
        self.lr_decay_rate = 0.005

        self.mlflow_logger = MLFlowLogger(experiment_name="imaging_model", run_name="first_run")
        mlflow.start_run()
        self.mlflow_logger.log_hyperparams({
            'l2_reg': l2_reg,
            'lr': self.lr,
            'lr_decay_rate': self.lr_decay_rate
        })
        
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        data_dict = patient_info_list
        train_files, val_files = train_test_split(data_dict, test_size=0.2, random_state=42)

        # Set the seed again?
        set_determinism(seed=42)

        # Resample images using median spacing
        '''
        We will use (or at least should, but dont get the formula yet) WL + (WW/2) for the upper grey level and WL - (WW/2) for the lower level
        I will use the values for soft tissue in the abdomen for now, so -125 to +225
        '''

        # Define transforms
        train_transforms = Compose([
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Spacingd(
                keys=['img'],
                pixdim=median_spacing.tolist(),
                mode=('bilinear'),
            ),
            ScaleIntensityRanged(
                keys=["img"],
                a_min=-125,
                a_max=225,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(
                keys=["img"],
                spatial_size=tuple(max_image_size),
                mode="minimum",
            ),
        ])

        val_transforms = Compose([
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Spacingd(
                keys=['img'],
                pixdim=median_spacing.tolist(),
                mode=('bilinear'),
            ),
            ScaleIntensityRanged(
                keys=["img"],
                a_min=-125,
                a_max=225,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(
                keys=["img"],
                spatial_size=tuple(max_image_size),
                mode="minimum",
            ),
        ])

        self.train_ds = Dataset(data = train_files, transform = train_transforms)
        self.val_ds = Dataset(data = val_files, transform = val_transforms)
        # Print the lengths of your training and validation datasets
        print("Length of training dataset:", len(self.train_ds))
        print("Length of validation dataset:", len(self.val_ds))


    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_ds, batch_size=1, shuffle=True)  # Als alles dezelfde size is, kan dit groter
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_ds, batch_size=1)
        return val_dataloader
    
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
         # Print the shapes of risk_pred, y, and e
        print("Shape of risk_pred:", risk_pred.shape)
        print("Shape of y:", y.shape)
        print("Shape of e:", e.shape)

        # Print the values of risk_pred, y, and e
        print("Risk predictions:", risk_pred)
        print("Event durations:", y)
        print("Event indicators:", e)

        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy()
        if not isinstance(risk_pred, np.ndarray):
            risk_pred = risk_pred.detach().cpu().numpy()
        if not isinstance(e, np.ndarray):
            e = e.detach().cpu().numpy()

        return concordance_index(y, risk_pred, e) # Risk_pred should have a negative sign

    def training_step(self, batch, batch_idx):
        images, y, e = batch['img'], batch['duration'], batch['event']
        
        # Inside the training and validation step methods, print the shapes of inputs
        print("Shape of images:", images.shape)
        print("Shape of durations:", y.shape)
        print("Shape of events:", e.shape)

        risk_pred = self.forward(images)
        batch_dictionary = {
            'duration': y,
            'event': e,
            'risk_pred': risk_pred
        }
        self.training_step_outputs.append(batch_dictionary)

        return batch_dictionary

    def on_train_epoch_end(self):
        # Initialize lists to store predictions and ground truth values for all batches
        all_risk_pred = []
        all_y = []
        all_e = []

        # Aggregate predictions and ground truth over all batches in the epoch
        for batch in self.training_step_outputs:
            all_risk_pred.append(batch['risk_pred'])
            all_y.append(batch['duration'])
            all_e.append(batch['event'])

        # Inside the training and validation epoch end methods, print the lengths of lists
        print("Length of all_risk_pred:", len(all_risk_pred))
        print("Length of all_y:", len(all_y))
        print("Length of all_e:", len(all_e))

        # Concatenate predictions and ground truth values along the batch dimension
        aggregated_risk_pred = torch.cat(all_risk_pred, dim=0)
        aggregated_y = torch.cat(all_y, dim=0)
        aggregated_e = torch.cat(all_e, dim=0)

        # Compute loss using aggregated values
        epoch_loss = self.loss_fn(aggregated_risk_pred, aggregated_y, aggregated_e)

        # # Calculate C-index for the epoch
        # c_index_epoch = self.c_index(-aggregated_risk_pred, aggregated_y, aggregated_e)

        # Log aggregated loss and C-index for the epoch
        # self.mlflow_logger.log_metrics({'train_c_index': c_index_epoch})
        self.mlflow_logger.log_metrics({'train_loss': epoch_loss.item()})
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        images, y, e = batch['img'], batch['duration'], batch['event']
        risk_pred = self.forward(images)
        batch_dictionary = {
            'duration': y,
            'event': e,
            'risk_pred': risk_pred
        }
        self.validation_step_outputs.append(batch_dictionary)

        return batch_dictionary

    def on_validation_epoch_end(self):
        # Initialize lists to store predictions and ground truth values for all batches
        all_risk_pred = []
        all_y = []
        all_e = []

        # Aggregate predictions and ground truth over all batches in the epoch
        for batch in self.validation_step_outputs:
            all_risk_pred.append(batch['risk_pred'])
            all_y.append(batch['duration'])
            all_e.append(batch['event'])

        # Concatenate predictions and ground truth values along the batch dimension
        aggregated_risk_pred = torch.cat(all_risk_pred, dim=0)
        aggregated_y = torch.cat(all_y, dim=0)
        aggregated_e = torch.cat(all_e, dim=0)

        # Compute loss using aggregated values
        epoch_loss = self.loss_fn(aggregated_risk_pred, aggregated_y, aggregated_e)

        # # Calculate C-index for the epoch
        # c_index_epoch = self.c_index(-aggregated_risk_pred, aggregated_y, aggregated_e)

        # Log aggregated loss and C-index for the epoch
        # self.mlflow_logger.log_metrics({'val_c_index': c_index_epoch})
        self.mlflow_logger.log_metrics({'val_loss': epoch_loss.item()})
        self.validation_step_outputs.clear()  # free memory

# ---------------------------------------------------------------------------------------------------------------------------------
max_epochs = 500
l2_reg = 0
model = SurvivalImaging(l2_reg)

# Start the trainer
trainer = pl.Trainer(
    max_epochs = max_epochs,
    logger = model.mlflow_logger,
    accelerator = 'gpu',
    devices = 1,
)

trainer.fit(model)


