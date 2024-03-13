'''
This is script is meant to inspect my data to ensure that it is suitable for survival analysis.
It will not be used for actual analysis, but to inspect my data more thoroughly 
'''

# Imports
import h5py
import pandas as pd
from lifelines import KaplanMeierFitter
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
filepath = 'my_models/simple_model.h5'
train_df, test_df, combined_df = load_data_from_h5(filepath)
print(combined_df)

# Count occurrences of 'e' in the dataset
e_counts = combined_df['e'].value_counts()
print("The event occurance frequency is:")
print(e_counts)

kmf = KaplanMeierFitter()
kmf.fit(combined_df.loc['train']['y'], event_observed=combined_df.loc['train']['e'], label='Train')
kmf.plot_survival_function()
kmf.fit(combined_df.loc['test']['y'], event_observed=combined_df.loc['test']['e'], label='Test')
kmf.plot_survival_function()

plt.title('Survival curves patients with available images')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.savefig('/trinity/home/r098372/pycox/figures/survival_curves_159.png')
plt.show()

# Make an new figure
plt.figure()

plt.hist(combined_df['y'], bins=30, edgecolor='black')  # You can adjust the number of bins as needed
plt.title('Distribution of Survival Times of patients with images')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.savefig('/trinity/home/r098372/pycox/figures/distribution_y_159.png')
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------
# 1.2 All patients, also without available images
filepath = 'my_models/simple_model_all.h5'
train_df, test_df, combined_df = load_data_from_h5(filepath)
print(combined_df)
combined_df = combined_df.fillna(0)     # Needs to be fixed!

# Set the threshold and remove everyone above that threshold (this cannot be true)
threshold_value = 20000
combined_df = combined_df[combined_df['y'] <= threshold_value]

# Count occurrences of 'e' in the dataset
e_counts = combined_df['e'].value_counts()
print("The event occurance frequency is:")
print(e_counts)

# Lets make a new figure 
plt.figure()

kmf = KaplanMeierFitter()
kmf.fit(combined_df.loc['train']['y'], event_observed=combined_df.loc['train']['e'], label='Train')
kmf.plot_survival_function()
kmf.fit(combined_df.loc['test']['y'], event_observed=combined_df.loc['test']['e'], label='Test')
kmf.plot_survival_function()

plt.title('Survival curves all patients')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.savefig('/trinity/home/r098372/pycox/figures/survival_curves_1531.png')
plt.show()

# Make an new figure
plt.figure()

plt.hist(combined_df['y'], bins=30, edgecolor='black')  # You can adjust the number of bins as needed
plt.title('Distribution of Survival Times of all patients')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.savefig('/trinity/home/r098372/pycox/figures/distribution_y_1531.png')
plt.show()

'''
From these results we can see that the y values for patients without images are too high. 
We need to find the cause for this. It could obviously also influence our model, which is not something we actually want. 
So we need to go back to data processing. We can then also make the RFS models for the next step.

'''

# I will now do my RFS models
# 2.1 Patients with available images

# 2.2 All patients, also without available images

