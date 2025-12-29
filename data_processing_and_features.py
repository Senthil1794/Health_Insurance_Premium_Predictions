# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:24:33 2025

@author: Senthil
"""
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# fetch data from data base
def fetch_data_from_db(database, table):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    '''
    cursor.execute("Select name from sqlite_master where type='table';")
    tables = cursor.fetchall()
    print(tables)
    '''
    query = "select * from " + table
    data = pd.read_sql_query(query, conn)
    return data

# Find missing value in dataset and fill those value   
def missing_value_imputation(data, numerical_columns, categorical_columns):
    for column in numerical_columns :
        data[column].fillna(data[column].median(),inplace=True)
    for column in categorical_columns :
        data[column].fillna(data[column].mode()[0],inplace=True)
    return data

# Performs outlier treatment
def outlier_removal(data,numerical_columns):
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        print(Q1,Q3)
        data = data[(data[column] >= Q1 - 1.5*IQR)&(data[column] <= Q3+1.5*IQR)]
        print(IQR)
        return data

def chi2_test(data, categorical_columns, label_column):
    drop_feature_list = []
    for column in categorical_columns:
        table = pd.crosstab(data[column], data[label_column])
        print(table)
        stat, p, dof, expected = chi2_contingency(table)
        print("Chi-square test for feature: ", column)
        print("p-value : ", p)
        print("DOF: ", dof)
        print("")
        if p > 0.5:
            drop_feature_list.append(column)
        else:
            pass
    return drop_feature_list

def train_test_split_features(data,label) :
    y = data[label]
    x = data.drop(label,axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle = False, random_state = 0)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test, features

def fit_and_evaluate_model(x_train, x_test, y_train, y_test,model,feature):
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Optional: Feature importance
    print("\nFeature Importances:")
    for i, importance in enumerate(model.feature_importances_):
        print(f"Feature {feature[i]}: {importance:.4f}")
    

            
    




        
    


