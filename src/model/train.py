# Databricks notebook source
import mlflow
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
mlflow.set_registry_uri("databricks-uc")

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

tags = {
    "model_type": "xgboost",
    "ai_product_lead": "Brandon.Jenkins@imail.org",
    "ai_developer": "Brandon.Jenkins@imail.org",
    "mlops_engineer": "mark.nielsen@imail.org",
    "business_steward": "Andy.Merrill@imail.org",
    "data_architect": "Madison.Johnson3@imail.org",
    "target_type": "classification",
}

# COMMAND ----------

df = spark.sql('''
    select
    -- encounter_id,
    -- patient_nbr,
    number_inpatient,
    discharge_disposition_id,
    -- admission_source_id,
    number_diagnoses,
    -- diabetesMed,
    number_emergency,
    number_outpatient,
    weight,
    payer_code,
    age,
    admission_type_id,
    -- medical_specialty,
    -- diag_1,
    race,
    num_procedures,
    case when readmitted = '<30' then 1
        when readmitted = '>30' then 1
        else 0 end as label
    from datascience.ds_workbench.uci_diabetic_data_train
''')

test_data = spark.sql('''
  select
    -- encounter_id,
    -- patient_nbr,
    number_inpatient,
    discharge_disposition_id,
    -- admission_source_id,
    number_diagnoses,
    -- diabetesMed,
    number_emergency,
    number_outpatient,
    weight,
    payer_code,
    age,
    admission_type_id,
    -- medical_specialty,
    -- diag_1,
    race,
    num_procedures,
    case when readmitted = '<30' then 1
        when readmitted = '>30' then 1
        else 0 end as label
  from datascience.ds_workbench.uci_diabetic_data_test
                      ''')

# COMMAND ----------

# create empty lists to store the transformers and columns
indexers = []
encoders = []
cols = []

# loop through each column in the dataframe
for col in df.columns:
# check if the column is categorical
    if col in ['discharge_disposition_id', 'admission_source_id', 'diabetesMed', 'weight', 'payer_code',
               'age', 'admission_type_id', 'medical_specialty', 'diag_1', 'race']:
        # create a StringIndexer object and fit it to the column
        indexer = StringIndexer(inputCol=col, outputCol=col+"_index")
        indexers.append(indexer)
    
        if col in ['discharge_disposition_id', 'admission_source_id', 'weight', 'payer_code',
                'age', 'admission_type_id', 'medical_specialty', 'diag_1', 'race']:
            # create a OneHotEncoder object and fit it to the indexed column
            encoder = OneHotEncoder(inputCol=col+"_index", outputCol=col+"_vec")
            encoders.append(encoder)
            
            # add the encoded column to the list of columns
            cols.append(col+"_vec")
        else:
            cols.append(col+"_index")
else:
    # add the original column to the list of columns
    cols.append(col)

# create a VectorAssembler object to combine the encoded features into a single feature vector
assembler = VectorAssembler(inputCols=cols, outputCol="features")

# create an XGBoostClassifier object with the desired hyperparameters
xgb = SparkXGBClassifier(maxDepth=3, numRound=10)

# create a pipeline to chain the transformers and estimator together
pipeline = Pipeline(stages=indexers + encoders + [assembler, xgb])

# fit the pipeline to the data
model = pipeline.fit(df)

# COMMAND ----------

predictions = model.transform(test_data)
# Evaluate the model using accuracy metric
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

# COMMAND ----------

input_schema = Schema(
    [
        ColSpec("integer", "number_inpatient"),
        ColSpec("string", "discharge_disposition_id"),
        ColSpec("integer", "number_diagnoses"),
        ColSpec("integer", "number_emergency"),
        ColSpec("integer", "number_outpatient"),
        ColSpec("string", "weight"),
        ColSpec("string", "payer_code"),
        ColSpec("string", "age"),
        ColSpec("integer", "admission_type_id"),
        ColSpec("string", "race"),
        ColSpec("integer", "num_procedures"),
    ]
)
output_schema = Schema([ColSpec("integer")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run():
    mlflow.log_artifacts("../../model-card/", artifact_path="model/model-card")
    mlflow.set_tags(tags)

    mlflow.spark.log_model(
        model,
        artifact_path="model",
        registered_model_name="datascience.ds_output.ai-product-example",
        signature=signature)

    mlflow.log_metric("accuracy", accuracy)

# COMMAND ----------

accuracy

# COMMAND ----------

# # Loop through columns in test data and check if categories are present in training data
# for col in test_data.columns:
#     train_categories = set(df.select(col).distinct().rdd.flatMap(lambda x: x).collect())
#     test_categories = set(test_data.select(col).distinct().rdd.flatMap(lambda x: x).collect())
#     if not test_categories.issubset(train_categories):
#         print("Test data has new categories in column '{}' that are not present in the training data.".format(col))
