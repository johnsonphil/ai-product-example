# Databricks notebook source
import pandas as pd
df = pd.read_csv('diabetic_data.csv')
# df = df.replace(['?'],None)

# COMMAND ----------

sdf = spark.createDataFrame(df)

# COMMAND ----------

sdf.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from datascience.ds_workbench.uci_diabetic_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove '?' and split into train/test/prod

# COMMAND ----------

sdf_test, sdf_prod, sdf_train = sdf.randomSplit([0.2, 0.2, 0.6])
prod1, prod2, prod3, prod4, prod5, prod6, prod7 = sdf_prod.randomSplit([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])

# COMMAND ----------

sdf_train.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_train')
sdf_test.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_test')
prod1.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod1')
prod2.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod2')
prod3.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod3')
prod4.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod4')
prod5.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod5')
prod6.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod6')
prod7.write.mode('overwrite').saveAsTable('datascience.ds_workbench.uci_diabetic_data_prod7')
