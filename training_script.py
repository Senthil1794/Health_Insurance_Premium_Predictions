# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:14:58 2025

@author: Senthil
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from data_processing_and_features import (
    fetch_data_from_db,
    missing_value_imputation,
    outlier_removal,
    chi2_test,
    train_test_split_features,
    fit_and_evaluate_model
    )

data = fetch_data_from_db("regression.db", "Insurance_Prediction")

numerical_columns = ["age", "bmi", "children", "charges"]
categorical_columns = ["gender", "smoker", "region", "medical_history", "family_medical_history",
                       "exercise_frequency", "occupation", "coverage_level"]
categorical_columns1 = ["region", "medical_history", "family_medical_history",
                       "exercise_frequency", "occupation", "coverage_level"]
#print(data)
print(data.describe())
#print(data.isnull().sum())
data = missing_value_imputation(data,numerical_columns,categorical_columns)
data = outlier_removal(data, numerical_columns) 

for cols in categorical_columns:
    print(data[cols].value_counts())
    
data["gender"] = data["gender"].map({"male":0,"female":1})
data["smoker"] = data["smoker"].map({"yes":1,"no":0})

data = pd.get_dummies(data,columns=["region", "medical_history", "family_medical_history",
                       "exercise_frequency", "occupation", "coverage_level"],dtype=int, drop_first=True)


print(data.head())

y = data["charges"]
x = data.drop("charges",axis=1)

x_train, x_test, y_train, y_test, feature = train_test_split_features(data, "charges")
print(feature)

model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100, 
                         learning_rate=0.1, 
                         max_depth=5, 
                         random_state=42)
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test,model,feature)

