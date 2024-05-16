# Deep learning models
These are scripts used for model development. I will specify the purpose of each script
I will only describe the scripts that are actually still used. 

## Useful scripts, goal and outcome
- Model_jette.py
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Average test C-index and confidence interval and hyperparameters of all 5 outer folds (nested (inner=3, outer=5) k fold cross validation)
    - goal: Train deep learning network based on clinical features with hyperparameters optimization with Optuna (results on Teams for each model and Abstract...log)

- Model_jette_noopt.py
    - input: 
        - .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
        - Hyperparameters of the best performing outer fold of Model_jette.py
    - output: Average test C-index and confidence interval with 5 fold cross validation (not nested)
    - goal: Train deep learning network based on clinical features with hyperparameters found with hyperparameter optimization (results on Teams for each model and V2_simple...log)

- CPHmodel.py
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Average test C-index and confidence interval with 5 fold cross validation for CPH model and insight in what my model looks at (which params)
    - goal: Train a CPH model based on clinical features as comparison for my deep learning network

- Survival_curves.py 
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Kaplan Meier (Survival/RFS) curves
    - goal: Look into survival curves of my dataset, to see whether data is feasible to work with

# We can copy the following params for the models:
## Model 1 (Survival)
dim_2 = 99  # Example hyperparameter, you should choose based on prior knowledge or experimentation
dim_3 = 55
drop = 0.22541305037492282
l2_reg = 13.152435544780317

## Model 1 (RFS)
dim_2 = 95  # Example hyperparameter, you should choose based on prior knowledge or experimentation
dim_3 = 71
drop = 0.18436438636054864
l2_reg = 12.795963862534695

## Model 2 (Survival)
dim_2 = 100  # Example hyperparameter, you should choose based on prior knowledge or experimentation
dim_3 = 67
drop = 0.2741697615030259
l2_reg = 14.598141727220037

## Model 2 (RFS)
dim_2 = 100  # Example hyperparameter, you should choose based on prior knowledge or experimentation
dim_3 = 67
drop = 0.2741697615030259
l2_reg = 14.598141727220037

## The meaning of my features
# Simple model
# Clinical model
