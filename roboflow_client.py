import requests

from inference_sdk import InferenceHTTPClient

# Initialize client ONCE
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="DP9CNOSQHY2lZfpyJ0nI"
)

WORKSPACE = "coursemon"
WORKFLOW_ID = "detect-and-classify-2"

def detect_cow_features(image_path):
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        use_cache=True
    )

    if not result or not isinstance(result, list):
        return None

    run = result[0]

    detection = run.get("detection_predictions")
    if not detection:
        return None

    preds = detection.get("predictions", [])
    if not preds:
        return None

    cow = max(preds, key=lambda d: d.get("confidence", 0))

    cow_confidence = cow.get("confidence", 0)
    bbox = [
        cow.get("x", 0),
        cow.get("y", 0),
        cow.get("width", 0),
        cow.get("height", 0),
    ]

    disease_label = None
    disease_conf = 0

    cls = run.get("classification_predictions", [])
    if cls:
        pred = cls[0].get("predictions", {})
        disease_label = pred.get("top")
        disease_conf = pred.get("confidence", 0)

    return {
        "cow_confidence": cow_confidence,
        "bbox": bbox,
        "disease": disease_label,
        "disease_confidence": disease_conf,
    }
