import yaml
import os
import json
import joblib
import numpy as np


params_path = "params.yaml"
schema_path = os.path.join("prediction_service","schema_in.json")
with open(schema_path) as json_file:
    schema=json.load(json_file)
print(list(schema.keys()))