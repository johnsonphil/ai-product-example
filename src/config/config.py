# Databricks notebook source
import mlflow
mlflow.autolog(disable=True)
mlflow.set_registry_uri("databricks-uc")
MLFLOW_EXPERIMENT = "/Users/mark.nielsen@imail.org/ai-product-example"
mlflow.set_experiment(MLFLOW_EXPERIMENT)
