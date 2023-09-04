# Databricks notebook source
# MAGIC %pip install pytest

# COMMAND ----------

import pytest
from pathlib import Path
import sys
import os

import src

root_path = Path(src.__file__).resolve().parent.parent
os.chdir(root_path)

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# Run pytest.
retcode = pytest.main([".", "-v", "-p", "no:cacheprovider"])

# Fail the cell execution if there are any test failures.
assert retcode == 0, "The pytest invocation failed. See the log for details."
