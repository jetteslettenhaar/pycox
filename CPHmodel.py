'''
The cox proportional hazard model (CPH), so I can see which factores will influence survival the most
'''

# Imports
import h5py
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
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


'''
If we want to make KM plots we also want to run this second part!!!
'''

# # Voorspel de Progression Risk Scores voor de test dataset met het getrainde CPH-model
# predicted_risk_scores_test = cph.predict_partial_hazard(combined_df.loc['test'])

# # Bepaal de Median Progression Risk Score voor de test dataset
# median_progression_risk_score_test = np.median(predicted_risk_scores_test)

# # Definieer Low-Risk en High-Risk Groepen op basis van de mediane progressie risico score voor de test dataset
# risk_groups_test = np.where(predicted_risk_scores_test >= median_progression_risk_score_test, 'high-risk', 'low-risk')

# # Voeg de risico groepen toe aan de DataFrame voor de test dataset
# combined_df.loc['test', 'risk_group'] = risk_groups_test

# # Creëer KaplanMeierFitter object
# kmf = KaplanMeierFitter()

# # Fit het model en plot de Kaplan-Meier curve voor low-risk groep op de test dataset
# kmf.fit(combined_df.loc[('test', combined_df['risk_group'] == 'low-risk'), 'y'], 
#         event_observed=combined_df.loc[('test', combined_df['risk_group'] == 'low-risk'), 'e'], 
#         label='Low-Risk')
# ax = kmf.plot()

# # Fit het model en plot de Kaplan-Meier curve voor high-risk groep op de test dataset
# kmf.fit(combined_df.loc[('test', combined_df['risk_group'] == 'high-risk'), 'y'], 
#         event_observed=combined_df.loc[('test', combined_df['risk_group'] == 'high-risk'), 'e'], 
#         label='High-Risk')
# kmf.plot(ax=ax)

# # Voeg labels en legenda toe
# plt.xlabel('Time')
# plt.ylabel('Survival Probability')
# plt.title('Kaplan-Meier Curve (Simple model RFS)')
# plt.legend()
# plt.show()
# plt.savefig('/trinity/home/r098372/pycox/figures/KM_simple_RFS')