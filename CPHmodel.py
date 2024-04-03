'''
The cox proportional hazard model (CPH), so I can see which factores will influence survival the most
'''

# Imports
import h5py
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# Lets start with loading the data from the models

def load_data_from_h5(filepath):
    '''
    This function loads the data from my h5py file. The file contains the following information:
    X: covariates of the model
    e: whether the event (death or RFS) occurs? (1: occurs; 0: censored)
    t/y (t in this .h5 file): the time of event e.
    '''
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

    columns = ['x' + str(i) for i in range(x_train.shape[1])]

    train_df = (pd.DataFrame(x_train, columns=columns)
                    .assign(y=t_train)
                    .assign(e=e_train))

    test_df = (pd.DataFrame(x_test, columns=columns)
                   .assign(y=t_test)
                   .assign(e=e_test))

    # Combine train and test DataFrames
    combined_df = pd.concat([train_df, test_df], keys=['train', 'test'], names=['dataset'])

    return train_df, test_df, combined_df


# 1. I will start with my survival models
# 1.1 Patients with available images
filepath = 'my_models/simple_model_all.h5'
train_df, test_df, combined_df = load_data_from_h5(filepath)
print(combined_df)
combined_df = combined_df.dropna()

cph = CoxPHFitter(penalizer=0.0001)

# Fit the CoxPHFitter to your training data
cph.fit(combined_df.loc['train'], duration_col='y', event_col='e')

# After fitting, you can print out the summary of the fitted model
cph.print_summary()