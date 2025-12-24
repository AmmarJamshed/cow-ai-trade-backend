import requests
import base64
import os

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL = "cows-mien3-33bgs/1"  # change if your model version is different

def detect_cow_features(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://detect.roboflow.com/{MODEL}?api_key={ROBOFLOW_API_KEY}"

    response = requests.post(
        url,
        json={"image": encoded},
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    if response.status_code != 200:
        print("Roboflow error:", response.text)
        return None

    data = response.json()

    if "predictions" not in data or len(data["predictions"]) == 0:
        return None

    p = data["predictions"][0]

    return {
        "cow_confidence": p.get("confidence", 0),
        "bbox": [
            p.get("x", 0),
            p.get("y", 0),
            p.get("width", 0),
            p.get("height", 0),
        ],
        "disease": p.get("class", None)
    }
