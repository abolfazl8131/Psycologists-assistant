import mlflow
import os

mlflow.set_experiment("Psyco assistant run projects")
mlflow.set_tracking_uri(str(os.environ.get('MLFLOW_TRACKING_URI')))  
mlflow.projects.run(".", entry_point='main.sh')
