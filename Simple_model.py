"""
This is a script to run my simple model with the provided scripts. The script is saved locally in 'my_models' and is loaded differently from the other models.
Input: .h5 file
Output: Predictions on survival
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.datasets import simplemodel
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import pandas as pd
import h5py
from collections import defaultdict
np.random.seed(1234)
_ = torch.manual_seed(123)

# We are going to start by doing the same thing they are doing to their .h5 file but then to our own .h5 file

def load_and_process_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        data = defaultdict(dict)
        for ds in f:
            for array in f[ds]:
                data[ds][array] = f[ds][array][:]

    train = _make_df(data['train'])
    test = _make_df(data['test'])
    df = pd.concat([train, test]).reset_index(drop=True)
    return df

def _make_df(data):
    x = data['x']
    t = data['t']
    d = data['e']

    colnames = ['x' + str(i) for i in range(x.shape[1])]
    df = (pd.DataFrame(x, columns=colnames)
          .assign(duration=t)
          .assign(event=d))
    return df

# Example usage:
h5_file_path = 'my_models/simple_model.h5'  # Update with your file path
df_train = load_and_process_h5(h5_file_path)
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
print(df_train.head())


# Standardize
cols_standardize = ['x12', 'x13']
cols_leave = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x14', 'x15']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

# No label transform
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

# Neural network
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

# Training the model
model = CoxPH(net, tt.optim.Adam)
batch_size = 256
model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
print(log)

# Prediction
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_test)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
plt.xlabel('Time')

# Save the plot to a specific location (replace 'path/to/save/plot.png' with your desired file path)
plt.savefig('/trinity/home/r098372/pycox/output/survival_output_simple_model.png')

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
print('Concordence:\n', ev.concordance_td())
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
print('Brier score:\n', ev.integrated_brier_score(time_grid))
print('nbll:\n', ev.integrated_nbll(time_grid))