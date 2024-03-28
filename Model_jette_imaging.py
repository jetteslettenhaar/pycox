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
    # Make the labels floats, so the model can actually use them
    duration_str = patient_dict['duration']
    duration_str_without_days = duration_str.replace('days', '').strip()
    patient_dict['duration'] = float(duration_str_without_days)
    patient_dict['event'] = patient_dict['event'].astype(float)
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

class SurvivalImaging(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1
        )
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        data_dict = patient_info_list
        train_files, val_files = train_test_split(data_dict, test_size=0.2, random_state=42)

        # Set the seed again?
        set_determinism(seed=42)

        # Define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=['img']),
                EnsureChannelFirstd(keys=['img'])
                # Lets start without a spacing transform since we do not know the spacing yet
                # Spacingd(
                #     keys=['img'],
                #     pixdim=(1.0, 1.0, 1.0),
                #     mode=('bilinear'),
                # )
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=['img']),
                EnsureChannelFirstd(keys=['img'])
                # Lets start without a spacing transform since we do not know the spacing yet
                # Spacingd(
                #     keys=['img'],
                #     pixdim=(1.0, 1.0, 1.0),
                #     mode=('bilinear'),
                # )
            ]
        )

        self.train_ds = Dataset(data = train_files, transform = train_transforms)
        # Hier doorheen en kijken wat de spacing is (opslaan in een lijst samen met de size) zowel voor validatie als training, hiervoor spacing weghalen, kunt niet de size veranderen
        # Je kunt de size veranderen met padding tot grootste afbeelding --> minimale value in je beeld
        # Normalisatie van de intensiteit (Monai, scale intensity range d)

        self.val_ds = Dataset(data = val_files, transform = val_transforms)


shapes = []
spacings = []
# Assuming `patient_info_list` contains the patient information dictionaries
for patient_info in patient_info_list:
    # Load the image
    image_path = patient_info['img']
    image_data = nib.load(image_path)
    image = nib.load(image_path).get_fdata()

    # Get the shape
    image_shape = image.shape
    shapes.append(image_shape)
    # Get the voxel spacing
    voxel_spacing = np.array(image_data.header.get_zooms())
    spacings.append(voxel_spacing)

# Print the lists of shapes and spacings
print("List of image shapes:", shapes)
print("List of voxel spacings:", spacings)




    # def train_dataloader(self):
    #     train_dataloader = DataLoader(self.train_ds, batch_size=1, shuffle=True)  # Als alles dezelfde size is, kan dit groter
    #     return train_loader

    # def val_dataloader(self):
    #     val_dataloader = DataLoader(self.val_ds, batch_size=1)
    #     return val_dataloader
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=0.001)        
    #     def lr_lambda(epoch):
    #         lr_decay_rate = 0.1                                         
    #         return 1 / (1 + epoch * lr_decay_rate)                      
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    #     return [optimizer], [scheduler]

    # def loss_fn()
    #     mask = torch.ones(y.shape[0], y.shape[0]).to(device)
    #     mask[(y.permute(*torch.arange(y.ndim - 1, -1, -1)) - y) > 0] = 0
    #     log_loss = torch.exp(risk_pred) * mask
    #     log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
    #     log_loss = torch.log(log_loss).reshape(-1, 1)
    #     neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        
    #     # L2 regularization (not working now, we need Regularisation function!)
    #     if self.l2_reg > 0:
    #         l2_loss = self.regularization(self.model)
    #         return neg_log_loss + l2_loss
    #     else:
    #         return neg_log_loss
    
    # def c_index(self, risk_pred, y, e):
    #     '''
    #     We want to check whether the inputs are numpy arrays (this is expected in the function concordance_index).
    #     If not, we have to convert them to numpy arrays. Then we can use the imported function to calculate the c-index
    #     NOTETHAT this is now only using uncensored data (which is not a lot, especially for test set)
    #     '''
    #     if not isinstance(y, np.ndarray):
    #         y = y.detach().cpu().numpy()
    #     if not isinstance(risk_pred, np.ndarray):
    #         risk_pred = risk_pred.detach().cpu().numpy().squeeze()
    #     if not isinstance(e, np.ndarray):
    #         e = e.detach().cpu().numpy()

    #     return concordance_index(y, -risk_pred, e) # Risk_pred should have a negative sign

    # def training_step(self, batch, batch_idx):
    #     images, y, e = batch['img'], batch['duration'], batch['event']
    #     risk_pred = self.forward(images)
    #     loss = self.loss_fn(risk_pred, y, e)
    #     c_index = self.c_index(risk_pred, y, e)
    #     return {'loss': loss, 'c_index': c_index, 'risk_pred': risk_pred, 'y': y, 'e': e}
    
    # def validation_step(self, batch, batch_idx):
    #     images, y, e = batch['img'], batch['duration'], batch['event']



# # We need a regularization class just like in the other model to construct the loss function
# class Regularization(object):
#     def __init__(self, order, weight_decay):
#         ''' The initialization of Regularization class

#         :param order: (int) norm order number
#         :param weight_decay: (float) weight decay rate
#         '''
#         super(Regularization, self).__init__()
#         self.order = order
#         self.weight_decay = weight_decay

#     def __call__(self, model):
#         ''' Performs calculates regularization(self.order) loss for model.

#         :param model: (torch.nn.Module object)
#         :return reg_loss: (torch.Tensor) the regularization(self.order) loss
#         '''
#         reg_loss = 0
#         for name, w in model.named_parameters():
#             if 'weight' in name:
#                 reg_loss = reg_loss + torch.norm(w, p=self.order)
#         reg_loss = self.weight_decay * reg_loss
#         return reg_loss
