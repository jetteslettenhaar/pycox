'''
I need to make some tables for my report and I will make the script for that here
'''

# Imports
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set manual seed
seed = 42

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
filepath = 'my_models/simple_model_RFS_AGE.h5'
train_df, test_df, combined_df = load_data_from_h5(filepath)
print(combined_df)

threshold_value = 8760
combined_df = combined_df[combined_df['y'] <= threshold_value]
print(len(combined_df))

# train_df = combined_df.loc['train']

# # Select the columns x0 to x14
# selected_columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'e']

# # Initialize an empty dictionary to store the counts
# counts_per_column = {}

# # Loop through each column and count the occurrences of 1.0
# for column in selected_columns:
#     # Count the occurrences of 1.0 in the column
#     count = train_df[column].eq(1.0).sum()
#     # Store the count in the dictionary
#     counts_per_column[column] = count

# # Print the counts for each column
# for column, count in counts_per_column.items():
#     print(f"Count of 1.0 in column {column}: {count}")

# # Select the columns x15 and y
# new_columns = ['x12', 'y']

# # Calculate the mean and STD for each column
# for column in new_columns:
#     column_mean = train_df[column].mean()
#     column_std = train_df[column].std()
#     print(f"Mean of column {column}: {column_mean}")
#     print(f"Standard deviation of column {column}: {column_std}")


test_df = combined_df.loc['test']

# Select the columns x0 to x14
selected_columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'e']

# Initialize an empty dictionary to store the counts
counts_per_column = {}

# Loop through each column and count the occurrences of 1.0
for column in selected_columns:
    # Count the occurrences of 1.0 in the column
    count = test_df[column].eq(1.0).sum()
    # Store the count in the dictionary
    counts_per_column[column] = count

# Print the counts for each column
for column, count in counts_per_column.items():
    print(f"Count of 1.0 in column {column}: {count}")

# Select the columns x15 and y
new_columns = ['x12', 'y']

# Calculate the mean and STD for each column
for column in new_columns:
    column_mean = test_df[column].mean()
    column_std = test_df[column].std()
    print(f"Mean of column {column}: {column_mean}")
    print(f"Standard deviation of column {column}: {column_std}")


