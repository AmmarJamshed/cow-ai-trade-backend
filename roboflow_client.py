import requests
import base64
import os

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKFLOW_URL = "https://serverless.roboflow.com/coursemon/workflows/detect-and-classify-2"

def detect_cow_features(image_path):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_base64
            }
        }
    }

    response = requests.post(WORKFLOW_URL, json=payload, timeout=30)

    if response.status_code != 200:
        print("Roboflow error:", response.text)
        return None

    result = response.json()

    try:
        detections = result["outputs"][0]["predictions"]
        if not detections:
            return None

        cow = detections[0]

        return {
            "cow_confidence": cow["confidence"],
            "bbox": [
                cow["x"],
                cow["y"],
                cow["width"],
                cow["height"]
            ],
            "disease": cow.get("class", None)
        }

    except Exception as e:
        print("Parse error:", e)
        return None
