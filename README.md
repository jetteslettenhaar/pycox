# Deep learning models
These are scripts used for model development. I will specify the purpose of each script
I will only describe the scripts that are actually still used. 

## Useful scripts, goal and outcome
- Model_jette.py
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Average test C-index and confidence interval and hyperparameters of all 5 outer folds (nested (inner=3, outer=5) k fold cross validation)
    - goal: Train and evaluate deep learning network based on clinical features with hyperparameters optimization with Optuna 

- Model_jette_noopt.py
    - input: 
        - .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
        - Hyperparameters of the best performing outer fold of Model_jette.py
    - output: Average test C-index and confidence interval with 5 fold cross validation (not nested), high/low risk KM curve, Shapley analysis
    - goal: Train and evaluate deep learning network based on clinical features with hyperparameters found with hyperparameter optimization

- CPHmodel.py
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Average test C-index and confidence interval with 5 fold cross validation for CPH model and insight in what my model looks at (which params)
    - goal: Train a CPH model based on clinical features as comparison for my deep learning network

- Survival_curves.py 
    - input: .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
    - output: Kaplan Meier (Survival/RFS) curves
    - goal: Look into survival curves of my dataset, to see whether data is feasible to work with

- Model_jette_imaging.py
    - input: images saved (Look at GIST_dataverzameling map) and .h5 file containing outcomes for death/RFS
    - output: Validation C-index for original train/test split
    - goal: Train and evaluate deep learning model based on imaging 

- Remake_imaging_model.py
    - input: 
        - .h5 file containing features and outcomes for simple death/RFS and clinical death/RFS
        - Hyperparameters of the best performing outer fold of Model_jette.py
    - output: Average test C-index and confidence interval with 5 fold cross validation (not nested)
    - goal: To see whether small changes that were made in model architecture from clinical to imaging model were the cause of bad performance 

- Classification_model.py
    - input: images saved (Look at GIST_dataverzameling map on data/scratch/r098372/beelden...) and .h5 file containing outcomes for death/RFS
    - output: AUC and accuracy of the classification model into low- and high-risk group
    - goal: Train and evaluate classification model based on imaging

- Grad_CAM.py
    - input: Classification model for one specific train/test split
    - output: Saved classification model for one specific train/test split (survival_imaging_model.pth)
    - goal: Save model for Grad_CAM analysis

- Grad_CAM_V2.py
    - input: three images of which Grad_CAM is computed (data/scratch/r098372/beelden...) and model saved .pth (Grad_CAM.py)
    - output: Grad_CAM analysis of three images 
    - goal: Look into interpretability of the classification model


## Not so very useful scripts anymore
- Make_tables.py
    - Was made to make tables for my report with characteristics of my dataset

- Model_jette_test_img.py, Model_jette_test.py, Test_data.py and Test_on_imaging.py
    - Not entirely sure anymore what I wanted to do here, but I think I wanted to test things?

- Clinical_model.py and Simple_model.py
    - Was basically copied from online PyTorch version of DeepSurv, but did not actually work on my data 

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

