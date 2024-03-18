# Lets start again by importing the important things
import torch
from torch import nn
import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from monai.networks.nets import ResNet

# Get data ready and turn data into tensors
Image_path = '/data/scratch/r098372/beelden'

# Also need to import all the labels for these patients specifically 
File_path = 'my_models/outcome_model_imaging_v4.pickle'

with open(File_path, 'rb') as handle:
    labels = pickle.load(handle)

print(labels)
