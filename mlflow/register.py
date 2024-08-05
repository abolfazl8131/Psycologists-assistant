import mlflow
from transformers import pipeline,AutoTokenizer
import os

mlflow.set_experiment("Psyco assistant model registring process")
mlflow.set_tracking_uri(str(os.environ.get('MLFLOW_TRACKING_URI')))  

architecture = "psyco_assistant"
qa_pipe = pipeline(task="text-generation", tokenizer=AutoTokenizer.from_pretrained(architecture),model=architecture)


with mlflow.start_run() as run:
    
    mlflow.transformers.log_model(
        transformers_model=qa_pipe,
        artifact_path="psyco_assistant_artifacts",
    )

try:
    mlflow.register_model(f"runs:/{run.info.run_id}/psyco_assistant_artifacts", "psyco_tiny")
except Exception as e:
    print(e.message)