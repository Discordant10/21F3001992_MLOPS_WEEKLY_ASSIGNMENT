import vertexai
from vertexai.tuning import sft

PROJECT_ID = "project-ccb283b3-613f-40a9-a23"
REGION = "us-central1"
BUCKET_NAME = "irisdata-raw"

vertexai.init(project=PROJECT_ID, location=REGION)

print("Submitting v1 tuning job...")
v1_job = sft.train(
    source_model="gemini-1.5-flash-001",
    train_dataset=f"gs://{BUCKET_NAME}/v1_train.jsonl",
    epochs=3,
    learning_rate_multiplier=1.0,
    experiment="iris-v1",
    tuned_model_display_name="iris-v1-raw",
)

print("Submitting v2 tuning job...")
v2_job = sft.train(
    source_model="gemini-1.5-flash-001",
    train_dataset=f"gs://{BUCKET_NAME}/v2_train.jsonl",
    epochs=3,
    learning_rate_multiplier=1.0,
    experiment="iris-v2",
    tuned_model_display_name="iris-v2-natural",
)

print("Jobs submitted. Monitor them in the Vertex AI Console.")
