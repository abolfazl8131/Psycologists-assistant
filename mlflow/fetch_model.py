from mlflow import MlflowClient
import string
import random

client = MlflowClient()
def assign_alias_to_stage(model_name, stage):
    alias = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
    latest_mv = client.get_latest_versions(model_name, stages=[stage])[0]
    client.set_registered_model_alias(model_name, alias, latest_mv.version)


assign_alias_to_stage("pysco_tiny",'Production')