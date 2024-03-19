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


# Function to load NIFTI files for a given subject folder
def load_nifti_files(subject_folder):
    nifti_files = []
    # Traverse the subject folder to find NIFTI files
    for root, dirs, files in os.walk(subject_folder):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Construct the full path to the NIFTI file
                nifti_path = os.path.join(root, file)
                # Load the NIFTI file using nibabel
                img = nib.load(nifti_path)
                # Convert the NIFTI data to a torch tensor
                img_data = torch.tensor(img.get_fdata(), dtype=torch.float32)
                # Append the loaded NIFTI tensor to the list
                nifti_files.append(img_data)
    return nifti_files

# Function to load NIFTI files for all subjects
def load_all_nifti_files(image_path):
    all_nifti_files = {}
    subject_names = []  # List to store subject names
    # Traverse the root folder to find subject folders
    for subject_folder in os.listdir(image_path):
        subject_path = os.path.join(image_path, subject_folder)
        # Check if the path is a directory
        if os.path.isdir(subject_path):
            # Load NIFTI files for the current subject
            nifti_files = load_nifti_files(subject_path)
            # Store the NIFTI files in a dictionary with subject name as key
            all_nifti_files[subject_folder] = nifti_files
            subject_names.append(subject_folder)  # Append subject name
            print('I have done another subject')
    # Create a DataFrame with subject names and corresponding NIFTI tensors
    df = pd.DataFrame({'Subject_Name': subject_names, 'NIFTI_Tensors': list(all_nifti_files.values())})
    return df

image_path = '/data/scratch/r098372/beelden'
images_df = load_all_nifti_files(image_path)
print(images_df)

# Now lets put everything in one dataframe (both labels and nifti files)
labels_death = labels[['participant_id', 'Duration_death', 'Event_death']]
labels_RFS = labels[['participant_id', 'Duration_RFS', 'Event_RFS']]

# I need to remove the zero in the Subject_Name, otherwise they will not match very will
images_df['Subject_Name'] = images_df['Subject_Name'].str.replace(r'^0', '', regex=True)

# Now I can merge based on the participant so they have the correct labels
merged_death_df = pd.merge(images_df, labels_death, left_on='Subject_Name', right_on='participant_id', how='inner')
merged_RFS_df = pd.merge(images_df, labels_RFS, left_on='Subject_Name', right_on='participant_id', how='inner')
merged_death_df.drop(columns=['Subject_Name', 'participant_id'], inplace=True)
merged_RFS_df.drop(columns=['Subject_Name', 'participant_id'], inplace=True)

# Now you have two dataframes, merged_death_df and merged_RFS_df, containing images as tensors and labels
print("Merged DataFrame for death event:")
print(merged_death_df)

print("Merged DataFrame for RFS event:")
print(merged_RFS_df)
