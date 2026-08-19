import argparse
import json

import vertexai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vertexai.generative_models import GenerativeModel

# Replace these with your actual Endpoint IDs after training finishes
V1_ENDPOINT = "projects/project-ccb283b3-613f-40a9-a23/locations/us-central1/endpoints/YOUR_V1_ENDPOINT_ID"
V2_ENDPOINT = "projects/project-ccb283b3-613f-40a9-a23/locations/us-central1/endpoints/YOUR_V2_ENDPOINT_ID"

vertexai.init(project="project-ccb283b3-613f-40a9-a23", location="us-central1")


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def evaluate_model(endpoint, test_data, valid_outputs):
    model = GenerativeModel(endpoint)
    y_true, y_pred = [], []
    compliant_count = 0

    for item in test_data:
        expected = item["output_text"]
        try:
            response = model.generate_content(item["input_text"]).text.strip()
        except Exception:
            response = "API_ERROR"

        # Format Compliance Check
        if response in valid_outputs:
            compliant_count += 1
            y_pred.append(response)
        else:
            # If malformed, treat as an incorrect "unknown" class
            y_pred.append("malformed")

        y_true.append(expected)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    compliance_rate = compliant_count / len(test_data)

    return accuracy, precision, recall, compliance_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    v1_test = load_jsonl("v1_test.jsonl")
    v2_test = load_jsonl("v2_test.jsonl")

    valid_v1 = ["setosa", "versicolor", "virginica"]
    valid_v2 = [
        "This is Iris setosa.",
        "This is Iris versicolor.",
        "This is Iris virginica.",
    ]

    print("Evaluating v1 (Raw Format)...")
    v1_acc, v1_prec, v1_rec, v1_comp = evaluate_model(V1_ENDPOINT, v1_test, valid_v1)

    print("Evaluating v2 (Natural Format)...")
    v2_acc, v2_prec, v2_rec, v2_comp = evaluate_model(V2_ENDPOINT, v2_test, valid_v2)

    print(f"\n--- Results ---")
    print(
        f"v1 -> Acc: {v1_acc:.2f} | Prec: {v1_prec:.2f} | Rec: {v1_rec:.2f} | Compliance: {v1_comp:.2f}"
    )
    print(
        f"v2 -> Acc: {v2_acc:.2f} | Prec: {v2_prec:.2f} | Rec: {v2_rec:.2f} | Compliance: {v2_comp:.2f}"
    )

    if v1_acc < args.min_accuracy or v2_acc < args.min_accuracy:
        print(f"\nError: Model accuracy fell below the {args.min_accuracy} threshold.")
        exit(1)
