'''
We will start by preparing a dictionary.
This dictionary should be one dictionary for each patient with the image_path, duration and event 0/1.
All dictionaries of all patients should be placed in a list
'''

# Lets start again by importing the important things
import torch
from torch import nn
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import nibabel as nib

from monai.networks.nets import ResNet
from monai.transforms import (
    LoadImaged
    EnsureChannelFirstd
    Orientationd
    Spacingd
)

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
    print(patient_dict)



'''
The data has been prepared in a dictionary! 
Now we need to actually make the model which contains 'prepare_data' function to actually load the dictionary.
'''

# Let set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set manual seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class ResNet(pl.LightningModule):
    def __init__(self):
        self.model = Densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        

