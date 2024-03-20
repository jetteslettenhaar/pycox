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

# Import CSV file to get the labels of my images (pickle does not work with this (old) Python version)
labels = pd.read_csv('my_models/outcome_model_imaging.csv', delimiter=',')
labels = labels.drop(columns=['Censored_death', 'Censored_RFS'])
print(labels)

# Now lets put everything in one dataframe (both labels and nifti files)
labels_death = labels[['participant_id', 'Duration_death', 'Event_death']]
labels_RFS = labels[['participant_id', 'Duration_RFS', 'Event_RFS']]

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

    nifti_path = os.path.join(image_path, subject_name, 'NIFTI', os.listdir(os.path.join(image_path, subject_name, 'NIFTI'))[0])

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
