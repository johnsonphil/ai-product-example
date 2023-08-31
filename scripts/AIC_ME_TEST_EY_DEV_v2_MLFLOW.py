# Databricks notebook source
# MAGIC %md
# MAGIC # ML Demo Notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------

import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Dataset

# COMMAND ----------

#dbutils.fs.ls("/mnt/DS/ml_demo/diabetic_data.pq")
data = pd.read_parquet("/dbfs/mnt/DS/ml_demo/diabetic_data.pq")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explore Dataset
# MAGIC Understand Shape, Columns and Content in dataset

# COMMAND ----------

# Shape 
print("Row, Columns "+ str(data.shape) + '\n')

# COMMAND ----------

# Select top 15 input features + IDs + outcome 
selected_features = ['encounter_id', 'patient_nbr', 'number_inpatient', 'discharge_disposition_id', 'admission_source_id',
       'number_diagnoses', 'diabetesMed', 'number_emergency',
       'number_outpatient', 'weight', 'payer_code', 'age', 'admission_type_id',
       'medical_specialty', 'diag_1', 'race', 'num_procedures', 'readmitted']

data = data[selected_features]

# COMMAND ----------

data['readmitted'][data['readmitted']=='<30'] = 'YES' #class 1 for <30 readmission
data['readmitted'][data['readmitted']=='>30'] = 'YES' #class 2 for >30 readmission

# COMMAND ----------

data.head(5)

# COMMAND ----------

# Distribution of output column, 'readmitted'
data.readmitted.hist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Cleaning & Feature Engineering

# COMMAND ----------

# Check for missing values
# Represented as '?' in Dataset
missing_data = data.replace(['?'],None)
missing_data = missing_data.isnull().sum()
print(missing_data.sort_values(ascending=False))

# COMMAND ----------

# Dropping ID's since they are not applicable to the model
data.drop(columns=['encounter_id', 'patient_nbr'],inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocess Input and Output Columns for Model

# COMMAND ----------

# Encode all input columns to preprocess them for the model
# Needs an encoder dictionary if you want to be able to decode input values 

encoded_data = data.copy()
encoded_data.drop(columns=['readmitted'],inplace=True) 

encoded_data_cols = list(encoded_data.columns)

for col in encoded_data_cols:
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    encoded_data[col] = labelencoder.fit_transform(encoded_data[col])

# COMMAND ----------

# Encode output column to use in model
# Labels
# 0: <30
# 1: >30
# 2: NO
label_encoder_y = LabelEncoder()
y = data['readmitted']
label_encoder_y = label_encoder_y.fit(y)
label_encoded_y = label_encoder_y.transform(y)
print('Encoded Labels:',np.unique(label_encoded_y))
print('Decoded Labels:',label_encoder_y.inverse_transform(np.unique(label_encoded_y)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model: XGBoost 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Autolog Experiment

# COMMAND ----------

import mlflow
# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
#mlflow.sklearn.autolog() # wont work since we're using xgboost and not sklearn
# https://www.mlflow.org/docs/latest/python_api/mlflow.xgboost.html#mlflow.xgboost.autolog
mlflow.xgboost.autolog()


# Use encoded input and output to put through model
X = encoded_data
y = label_encoded_y
print('Features:', X.shape[1])
print('Label Shape:',y.shape)

# Split data into training/testing subsets (75/25) split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tags = {"product_version": '1.0',
        "model_type": 'XGBoost Classification'}

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run(run_name='XGBoost Run'):

    model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric=['logloss','error','auc','aucpr'])
    model.fit(X_train, y_train,eval_set = [(X_test, y_test)])
    predictions = model.predict(X_test)
    mlflow.log_artifacts('../model-card/', artifact_path='model-card')
    mlflow.set_tags(tags)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rest of End to End Examples pending on requirements... (model serving, hyperparamter tuning etc)

# COMMAND ----------


