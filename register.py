import mlflow
from transformers import pipeline
import os


mlflow.set_experiment("Psyco assistant Expriment")
mlflow.set_tracking_uri(str(os.environ['HOST']))  

architecture = "psyco_assistant"
qa_pipe = pipeline("question-answering", architecture)

with mlflow.start_run():
    
    mlflow.transformers.log_model(
        transformers_model=qa_pipe,
        artifact_path="psyco_assistant_artifacts",
    )