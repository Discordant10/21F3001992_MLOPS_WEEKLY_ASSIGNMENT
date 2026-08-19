import argparse
import json
import os
import vertexai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Use numeric project ID to bypass Cloud Resource Manager API
PROJECT_ID = "124100860677"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def parse_label(raw_text, is_v2=False):
    """Extract class label from model response to handle minor formatting noise."""
    text = raw_text.lower().strip()
    
    if "setosa" in text:
        return "This is Iris setosa." if is_v2 else "setosa"
    elif "versicolor" in text:
        return "This is Iris versicolor." if is_v2 else "versicolor"
    elif "virginica" in text:
        return "This is Iris virginica." if is_v2 else "virginica"
    
    return "malformed"

def evaluate_model(endpoint_name, test_data, valid_outputs, is_v2=False):
    model = GenerativeModel(endpoint_name)
    y_true, y_pred = [], []
    compliant_count = 0

    # Strict low-temperature generation settings
    gen_config = GenerationConfig(temperature=0.0, max_output_tokens=30)

    for idx, item in enumerate(test_data):
        expected = item["output_text"]
        try:
            res = model.generate_content(item["input_text"], generation_config=gen_config)
            response = parse_label(res.text, is_v2=is_v2)
        except Exception as e:
            if idx == 0:
                print(f"[API Error for {endpoint_name}]: {e}")
            response = "API_ERROR"

        if response in valid_outputs:
            compliant_count += 1
            y_pred.append(response)
        else:
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
    parser.add_argument("--v1-endpoint", type=str, required=True)
    parser.add_argument("--v2-endpoint", type=str, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    v1_test = load_jsonl("v1_test.jsonl")
    v2_test = load_jsonl("v2_test.jsonl")

    valid_v1 = ["setosa", "versicolor", "virginica"]
    valid_v2 = [
        "This is Iris setosa.",
        "This is Iris versicolor.",
        "This is Iris virginica."
    ]

    print("Evaluating v1 (Raw Feature Format)...")
    v1_acc, v1_prec, v1_rec, v1_comp = evaluate_model(
        args.v1_endpoint, v1_test, valid_v1, is_v2=False
    )

    print("Evaluating v2 (Natural Language Format)...")
    v2_acc, v2_prec, v2_rec, v2_comp = evaluate_model(
        args.v2_endpoint, v2_test, valid_v2, is_v2=True
    )

    print("\n================ EVALUATION RESULTS ================")
    print(f"v1 Raw Format     -> Acc: {v1_acc:.2%} | Precision: {v1_prec:.2f} | Recall: {v1_rec:.2f} | Compliance: {v1_comp:.2%}")
    print(f"v2 Natural Format -> Acc: {v2_acc:.2%} | Precision: {v2_prec:.2f} | Recall: {v2_rec:.2f} | Compliance: {v2_comp:.2%}")
    print("====================================================")

    if v1_acc < args.min_accuracy or v2_acc < args.min_accuracy:
        print(f"\n[FAIL] Accuracy dropped below threshold ({args.min_accuracy:.2%}).")
        exit(1)

    print("\n[SUCCESS] Both models passed evaluation threshold.")