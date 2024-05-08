'''
The cox proportional hazard model (CPH), so I can see which factores will influence survival the most
'''

# Imports
import h5py
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
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
filepath = 'my_models/clinical_model_all_RFS_AGE.h5'
train_df, test_df, combined_df = load_data_from_h5(filepath)
print(combined_df)

threshold_value = 8760
combined_df = combined_df[combined_df['y'] <= threshold_value]
print(len(combined_df))

# Drop low variance columns
combined_df = combined_df.dropna()
threshold = 0.01  # Adjust the threshold as needed
variances = combined_df.var()
low_variance_columns = variances[variances < threshold].index
combined_df = combined_df.drop(columns=low_variance_columns)
print(combined_df)

cph = CoxPHFitter(penalizer=0.001)

# Fit the CoxPHFitter to your training data
cph.fit(combined_df.loc['train'], duration_col='y', event_col='e')

# After fitting, you can print out the summary of the fitted model
cph.print_summary()

# Perform k-fold cross validation
# Specify number of folds
num_folds = 5
scores = k_fold_cross_validation(cph, combined_df, duration_col='y', event_col='e', k=num_folds, scoring_method='concordance_index', seed=seed)

# Print the cross-validation scores
print("Cross-validation C-index scores:", scores)

# Calculate mean score
mean_score = np.mean(scores)
print("Mean C-index:", mean_score)

# Calculate 95% confidence interval
confidence_interval = 1.96 * np.std(scores) / np.sqrt(len(scores))
lower_bound = mean_score - confidence_interval
upper_bound = mean_score + confidence_interval
print("95% Confidence Interval:", (lower_bound, upper_bound))

# ---------------------------------------------------------------------------------------------------------------------------
'''
If we want to make KM plots we also want to run this second part!!!
'''

# Voorspel de Progression Risk Scores voor de test dataset met het getrainde CPH-model
predicted_risk_scores_test = cph.predict_partial_hazard(combined_df.loc['test'])

# Determine the percentiles of risk predictions
percentile_low = np.percentile(predicted_risk_scores_test, 25)
percentile_high = np.percentile(predicted_risk_scores_test, 75)

# Define thresholds for risk groups
threshold_low = percentile_low
threshold_high = percentile_high

# Categorize risk groups
risk_groups = np.where(predicted_risk_scores_test >= threshold_high, 'high-risk',
                    np.where(predicted_risk_scores_test <= threshold_low, 'low-risk', 'unknown/intermediate'))

# Initialize lists to store survival data for each risk group
high_risk_group = []
low_risk_group = []

# Split the test dataset into high-risk and low-risk groups
for i, risk_pred in enumerate(predicted_risk_scores_test):
    if risk_groups[i] == 'high-risk':
        high_risk_group.append((combined_df.loc['test'].iloc[i]['y'], combined_df.loc['test'].iloc[i]['e']))
    elif risk_groups[i] == 'low-risk':
        low_risk_group.append((combined_df.loc['test'].iloc[i]['y'], combined_df.loc['test'].iloc[i]['e']))

# Create KaplanMeierFitter objects
kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()

# Fit the models for high-risk and low-risk groups
y_high, e_high = zip(*high_risk_group)
y_low, e_low = zip(*low_risk_group)

# Fit the models for high-risk and low-risk groups
y_high = np.array(y_high)
e_high = np.array(e_high)
y_low = np.array(y_low)
e_low = np.array(e_low)

kmf_high.fit(y_high/365.5, e_high, label='High Risk Group')
kmf_low.fit(y_low/365.5, e_low, label='Low Risk Group')

# Plot the Kaplan-Meier curves
fig, ax = plt.subplots(figsize=(10, 6))
kmf_high.plot(ax=ax, color='red')
kmf_low.plot(ax=ax, color='green')

# Add at-risk counts
add_at_risk_counts(kmf_low, kmf_high, ax=ax)

# Set labels and legend
ax.set_xlabel('Time (years)')
ax.set_ylabel('Survival Probability')
ax.set_title('Kaplan-Meier Curve RFS')
ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))

# Set x-axis limit to 10 years
ax.set_xlim(0, 10)

plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('/trinity/home/r098372/pycox/figures/DSG_figures/KM_CPH_RFS')

# ---------------------------------------------------------------------------------------------------------------------------
'''
Now we want to show what happens with the survival when we have different groups (age, tumor size and primary tumor size)
FIRST WE DO THE SIMPLE MODEL!
'''

# # Define the age bins and labels
# age_bins = [0, 50, 65, 80, float('inf')]
# age_labels = ['Below 50', '50-65', '65-80', 'Above 80']

# # Divide the dataset into age groups
# combined_df['age_group'] = pd.cut(combined_df['x15'], bins=age_bins, labels=age_labels)

# # We want to use years, not days
# combined_df['y_years'] = combined_df['y']/365

# # Create a figure and axis object
# fig, ax = plt.subplots(figsize=(10, 6))

# # Creëer KaplanMeierFitter object
# kmf = KaplanMeierFitter()

# # Loop through each age group
# for age_group, group_data in combined_df.groupby('age_group'):
#     kmf.fit(group_data['y_years'], event_observed=group_data['e'], label=f'Age Group {age_group}')
#     kmf.plot(ax=ax)

# # Add labels, title, and legend
# plt.xlabel('Time (years)')
# plt.xlim(0, 10)
# plt.ylabel('Survival Probability')
# plt.title('Kaplan-Meier Curves for Different Age Groups (RFS)')
# plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/KM_different_groups/KM_simple_RFS_age_groups')

# # Now we want to make a histogram of the PRIMTUMSIZE
# # Create a figure and axis object
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot the histogram
# ax.hist(combined_df['x16'], bins=20, color='skyblue', edgecolor='black')

# # Add labels and title
# ax.set_xlabel('x16 Values')
# ax.set_ylabel('Frequency')
# ax.set_title('Histogram of x16')

# # Show the plot
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/histogramPRIMTUMSIZE')


# # Define the age bins and labels
# size_bins = [0, 50, 100, 150, float('inf')]
# size_labels = ['Below 50', '50-100', '100-150', 'Above 150']

# # Divide the dataset into size groups
# combined_df['size_group'] = pd.cut(combined_df['x16'], bins=size_bins, labels=size_labels)

# # We want to use years, not days
# combined_df['y_years'] = combined_df['y']/365

# # Create a figure and axis object
# fig, ax = plt.subplots(figsize=(10, 6))

# # Creëer KaplanMeierFitter object
# kmf = KaplanMeierFitter()

# # Loop through each age group
# for size_group, group_data in combined_df.groupby('size_group'):
#     kmf.fit(group_data['y_years'], event_observed=group_data['e'], label=f'Size group {size_group}')
#     kmf.plot(ax=ax)

# # Add labels, title, and legend
# plt.xlabel('Time (years)')
# plt.xlim(0, 10)
# plt.ylabel('Survival Probability')
# plt.title('Kaplan-Meier Curves for Different Primary Tumor Sizes (RFS)')
# plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/KM_different_groups/KM_simple_RFS_size_groups')

# ---------------------------------------------------------------------------------------------------------------------------
# '''
# Now we want to show what happens with the survival when we have different groups (age, tumor size and primary tumor size)
# NOW WE DO THE CLINICAL MODEL!
# '''

# # Now we want to make a histogram of the NUMMIT
# # Create a figure and axis object
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot the histogram
# ax.hist(combined_df['x37'], bins=20, range=(0,60), color='skyblue', edgecolor='black')

# # Add labels and title
# ax.set_xlabel('x37 Values')
# ax.set_ylabel('Frequency')
# ax.set_title('Histogram of x37')

# # Show the plot
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/histogramNUMMIT')

# # Define the mitoses bins and labels
# mit_bins = [0, 5, 10, 15, float('inf')]
# mit_labels = ['Below 5', '5-10', '10-15', 'Above 15']

# # Divide the dataset into size groups
# combined_df['mit_group'] = pd.cut(combined_df['x37'], bins=mit_bins, labels=mit_labels)

# # We want to use years, not days
# combined_df['y_years'] = combined_df['y']/365

# # Create a figure and axis object
# fig, ax = plt.subplots(figsize=(10, 6))

# # Creëer KaplanMeierFitter object
# kmf = KaplanMeierFitter()

# # Loop through each age group
# for mit_group, group_data in combined_df.groupby('mit_group'):
#     kmf.fit(group_data['y_years'], event_observed=group_data['e'], label=f'Mitoses group {mit_group}')
#     kmf.plot(ax=ax)

# # Add labels, title, and legend
# plt.xlabel('Time (years)')
# plt.xlim(0, 10)
# plt.ylabel('Survival Probability')
# plt.title('Kaplan-Meier Curves for Different Number of Mitoses (RFS)')
# plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/KM_different_groups/KM_simple_RFS_mit_group')
