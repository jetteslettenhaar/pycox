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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from monai.networks.nets import Densenet121
from monai.utils import set_determinism
from monai.data import Dataset
from monai.visualize.class_activation_maps import GradCAM
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    ScaleIntensityRanged,
)

import mlflow
from pytorch_lightning.loggers import MLFlowLogger

from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

import torchsummary

import h5py
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from torch.autograd import function
from torch.autograd import Variable
from scipy.ndimage import zoom
from PIL import Image


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
pl.seed_everything(seed)

# -------------------------------------------------------------------------------------------------------------------------------
'''
The data has been prepared in a dictionary! 
Now we need to actually make the model which contains 'prepare_data' function to actually load the dictionary.
'''


class SurvivalImaging(pl.LightningModule):
    def __init__(self, train_files, val_files):
        super().__init__()
        self.model = Densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )
        self.model.to(device)
        self.lr = 0.000001
        self.loss_function = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        self.train_files = train_files
        self.val_files = val_files

        self.mlflow_logger = MLFlowLogger(experiment_name="classification_model", run_name="GRADCAM")
        mlflow.start_run()
        self.mlflow_logger.log_hyperparams({
            'lr': self.lr
        })

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        # Set the seed again
        set_determinism(seed=42)

        # Define transforms
        transforms = Compose([
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

        self.train_ds = Dataset(data=self.train_files, transform=transforms)
        self.val_ds = Dataset(data=self.val_files, transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def accuracy_fn(self, y_true, y_pred):
        if len(y_pred) == 0:
            return 0  # Handle division by zero gracefully
        else:
            y_true_tensor = torch.tensor(y_true)
            y_pred_tensor = torch.tensor(y_pred)
            correct = torch.eq(y_true_tensor, y_pred_tensor).sum().item()
            acc = (correct / len(y_pred)) * 100
            return acc

    def training_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["risk_group"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        predictions = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)[:, 1]
        self.log('train_loss', loss, on_epoch=True)
        d = {"val_loss": loss, "predictions": predictions, "labels": labels, "probabilities": probabilities}
        self.training_step_outputs.append(d)
        return {"loss": loss, "predictions": predictions, "labels": labels, "probabilities": probabilities}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["risk_group"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        predictions = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)[:, 1]
        self.log('validation_loss', loss, on_epoch=True)
        d = {"val_loss": loss, "predictions": predictions, "labels": labels, "probabilities": probabilities}
        self.validation_step_outputs.append(d)
        return {"loss": loss, "predictions": predictions, "labels": labels, "probabilities": probabilities}

    def on_train_epoch_end(self):
        all_predictions = []
        all_labels = []
        all_probabilities = []
        for output in self.training_step_outputs:
            all_predictions.extend(output['predictions'].cpu().numpy())
            all_probabilities.extend(output['probabilities'].cpu().detach().numpy())
            all_labels.extend(output['labels'].cpu().numpy())

        all_predictions = np.array(all_predictions)
        print(all_predictions)
        all_probabilities = np.array(all_probabilities)
        print(all_probabilities)
        all_labels = np.array(all_labels)
        print(all_labels)

        accuracy = self.accuracy_fn(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_probabilities)
        self.log('train_accuracy', accuracy, on_epoch=True)
        self.log('train_roc', roc_auc, on_epoch=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        all_predictions = []
        all_labels = []
        all_probabilities = []
        for output in self.validation_step_outputs:
            all_predictions.extend(output['predictions'].cpu().numpy())
            all_probabilities.extend(output['probabilities'].cpu().detach().numpy())
            all_labels.extend(output['labels'].cpu().numpy())

        all_predictions = np.array(all_predictions)
        print(all_predictions)
        all_probabilities = np.array(all_probabilities)
        print(all_probabilities)
        all_labels = np.array(all_labels)
        print(all_labels)

        accuracy = self.accuracy_fn(all_labels, all_predictions)
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            roc_auc = 0.00
        self.log('val_accuracy', accuracy, on_epoch=True)
        self.log('val_roc', roc_auc, on_epoch=True)
        self.validation_step_outputs.clear()

    def on_train_end(self):
        mlflow.end_run()

# Train test split
data_dict = patient_info_list_filtered
train_files, test_files = train_test_split(data_dict, test_size=0.2, random_state=42)                 
max_epochs = 99                                          # Dit moet 99 zijn, nu voor testen even dit

model = SurvivalImaging(train_files, test_files)
model.load_state_dict(torch.load('survival_imaging_model.pth'))
model.to(device)

# Initialize GradCAM
gradcam = GradCAM(nn_module=model, target_layers="model.features.denseblock4.denselayer16.layers.conv2")
window_level = 40
window_width=  400

# Define image paths
image_paths = [
    '/data/scratch/r098372/beelden/101_1000/NIFTI/2_thxabd__50__b31f.nii.gz',
    '/data/scratch/r098372/beelden/101_1012/NIFTI/2_maagabdomen__30__b31f.nii.gz',
    '/data/scratch/r098372/beelden/101_1023/NIFTI/3_abdomen_maag__50__bf37__2.nii.gz'
]

# Define transforms
transforms = Compose([
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

# Create a figure for subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, image_path in enumerate(image_paths, start=1):
    # Apply the transforms to preprocess the image
    preprocessed_data = transforms({'img': image_path})
    
    # Convert preprocessed image to PyTorch tensor
    preprocessed_tensor = torch.tensor(preprocessed_data['img'], dtype=torch.float)
    
    # Add batch dimension if needed (assuming your model expects a batch of images)
    preprocessed_tensor = preprocessed_tensor.unsqueeze(0)

    input_tensor = preprocessed_tensor
    depth = input_tensor.shape[4]
    depth = depth + 20
    
    # Compute the heatmap using GradCAM for class 0 (low risk)
    heatmap_class_0 = gradcam(preprocessed_tensor.to(device), class_idx=0)
    
    print(f"Patient {idx} - Input tensor shape:", input_tensor.shape)
    print(f"Patient {idx} - Heatmap shape:", heatmap_class_0.shape)
    
    # Visualize the original image and the heatmap overlayed on the input image
    row = (idx - 1)
    
    # Original image
    axes[0, row].imshow(input_tensor.squeeze().cpu().numpy()[:, :, depth // 2], cmap='gray')
    axes[0, row].set_title(f'Patient {idx} - Original')

    # Heatmap for class 0
    im = axes[1, row].imshow(input_tensor.squeeze().cpu().numpy()[:, :, depth // 2], cmap='gray')
    im = axes[1, row].imshow(heatmap_class_0.squeeze().cpu().numpy()[:, :, depth // 2], alpha=0.5, cmap='jet')
    axes[1, row].set_title(f'Patient {idx} - GradCAM Class 0')
    
    # Add colorbar next to each heatmap
    cbar = plt.colorbar(im, ax=axes[1, row], fraction=0.046, pad=0.04)
    cbar.set_label('Heatmap Intensity')

# Adjust layout
plt.tight_layout()
plt.savefig('/trinity/home/r098372/pycox/figures/Classification/GradCAM_LowRisk_Subplots_7.png')
plt.close()

print("GradCAM visualizations for low risk patients have been saved.")