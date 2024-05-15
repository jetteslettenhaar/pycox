'''
We will start by preparing a dictionary.
This dictionary should be one dictionary for each patient with the image_path, duration and event 0/1.
All dictionaries of all patients should be placed in a list
'''

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

import torchsummary

import h5py
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

'''
 We need to start by actually making a classification between the two models. So we will use the CPH model and divide them into either low or high or intermediate risk.
 We use the values for low and high risk division from the CPHmodel trained on all samples.
'''
# Import CSV file to get the labels of my images (pickle does not work with this (old) Python version)
labels = pd.read_csv('my_models/outcome_model_imaging.csv', delimiter=',')
labels = labels.drop(columns=['Censored_death', 'Censored_RFS'])
print(labels)

# Now lets put everything in one dataframe (both labels and nifti files)
labels_death = labels[['participant_id', 'Duration_death', 'Event_death']]
labels_RFS = labels[['participant_id', 'Duration_RFS', 'Event_RFS']]
labels_death['Duration_death'] = labels_death['Duration_death'].str.replace(' days', '').astype(int)
labels_RFS['Duration_RFS'] = labels_RFS['Duration_RFS'].str.replace(' days', '').astype(int)
print(labels_death)
print(labels_RFS)

# Set manual seed
seed = 42

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
        subject_info = labels_RFS.merge(subject_df, on='participant_id', how='inner')
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
        'duration': subject_info['Duration_RFS'].values[0],
        'event': subject_info['Event_RFS'].values[0]
    }

    # Append this all to a list
    patient_info_list.append(patient_dict)                      

# Initialize risk groups dictionary
risk_groups = {'low-risk': 0, 'high-risk': 1}

# Print the list of patient dictionaries
for patient_dict in patient_info_list:
    patient_dict['duration'] = patient_dict['duration'].astype(float)
    patient_dict['event'] = patient_dict['event'].astype(float)
    # Initialize risk group
    risk_group = None
    # Check which risk group the participant belongs to
    if patient_dict['duration'] > 1825 and patient_dict['event'] == 0.0: 
        risk_group = 'low-risk'
    elif patient_dict['duration'] < 1825 and patient_dict['event'] == 1.0:
        risk_group = 'high-risk'
    
    # Add risk_group to patient_dict
    patient_dict['risk_group'] = risk_groups.get(risk_group, -1)  # Use .get() to handle cases where risk_group is not found

# Filteren op rows waar risk_group niet gelijk is aan -1
patient_info_list_filtered = [patient_dict for patient_dict in patient_info_list if patient_dict['risk_group'] != -1]

# Afdrukken van de gefilterde lijst van patiënten
for patient_dict in patient_info_list_filtered:
    print(patient_dict)

print(len(patient_info_list_filtered))

# Let set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set manual seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# -------------------------------------------------------------------------------------------------------------------------------
'''
The data has been prepared in a dictionary! 
Now we need to actually make the model which contains 'prepare_data' function to actually load the dictionary.
'''


class SurvivalImaging(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )
        self.model.to(device)
        self.lr = 0.01
        self.lr_decay_rate = 0.005
        self.loss_function = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

        self.mlflow_logger = MLFlowLogger(experiment_name="classification_model", run_name="run_1")
        mlflow.start_run()
        self.mlflow_logger.log_hyperparams({
            'lr': self.lr,
            'lr_decay_rate': self.lr_decay_rate
        })

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        data_dict = patient_info_list_filtered[:10]
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
                pixdim=(1, 1, 3),   # Using median in all direction resulted in GPU memory issue (CuDNN error)
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
                spatial_size=(32, 32, 32),
                mode="minimum",
            ),
        ])

        val_transforms = Compose([
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Spacingd(
                keys=['img'],
                pixdim=(1, 1, 3),
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
                spatial_size=(32, 32, 32),
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)        # Learning rate is hyperparameter! (normal is 0.001)

        def lr_lambda(epoch):
            lr_decay_rate = self.lr_decay_rate                          # Learning rate decay is a hyperparameter! (normal is 0.1)
            return 1 / (1 + epoch * lr_decay_rate)                      # Inverse time decay function using epoch like in DeepSurv

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]

    def accuracy_fn(self, y_true, y_pred):
        # Convert NumPy arrays to PyTorch tensors
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)

        # Use torch.eq() to compare tensors
        correct = torch.eq(y_true_tensor, y_pred_tensor).sum().item() 
        acc = (correct / len(y_pred)) * 100 
        return acc
    
    def training_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["risk_group"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        predictions = torch.argmax(output, dim=1)
        self.log('train_loss', loss, on_epoch=True)
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["risk_group"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        predictions = torch.argmax(output, dim=1)
        self.log('validation_loss', loss, on_epoch=True)
        d = {"val_loss": loss, "predictions": predictions, "labels": labels}
        self.validation_step_outputs.append(d)
        return {"loss": loss, "predictions": predictions, "labels": labels}
    
    def on_validation_epoch_end(self):
        # Collect predictions and ground truth labels from all validation batches
        all_predictions = []
        all_labels = []
        for output in self.validation_step_outputs:
            all_predictions.extend(output['predictions'].cpu().numpy())
            all_labels.extend(output['labels'].cpu().numpy())

        all_predictions = np.array(all_predictions)
        print(all_predictions)
        all_labels = np.array(all_labels)
        print(all_labels)

        # Calculate accuracy
        accuracy = self.accuracy_fn(all_labels, all_predictions)
        self.log('accuracy', accuracy, on_epoch=True)
        
    def on_train_end(self):
        mlflow.end_run()


# # ---------------------------------------------------------------------------------------------------------------------------------
max_epochs = 10
model = SurvivalImaging()

# Start the trainer
trainer = pl.Trainer(
    max_epochs = max_epochs,
    logger = model.mlflow_logger,
    accelerator = 'gpu',
    devices = 1,
    accumulate_grad_batches=2,
)

trainer.fit(model)

