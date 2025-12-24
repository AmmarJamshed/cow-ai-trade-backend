import os
from inference_sdk import InferenceHTTPClient

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

MODEL = "cows-mien3-33bgs/1"

def detect_cow_features(image_path):
    result = CLIENT.infer(image_path, model_id=MODEL)

    if "predictions" not in result or len(result["predictions"]) == 0:
        return None

    pred = result["predictions"][0]

    return {
        "cow_confidence": float(pred["confidence"]),
        "class": pred["class"]
    }
