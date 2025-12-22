from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import os
import hashlib

from roboflow_client import detect_cow_features
from similarity import calculate_similarity

app = Flask(__name__)

# CORS for Flutter Web / Mobile
CORS(app, resources={r"/*": {"origins": "*"}})

# Secret key
app.secret_key = "9b8f5b2c22f5731fb3584b6b411b291c74751c0223152e40fa7c8a98492cbd86"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global in-memory storage
stored_features = None
seller_image_hash = None


# ------------------------
# UTILS
# ------------------------
def image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ------------------------
# HOME (HTML UI)
# ------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ------------------------
# SELLER UPLOAD
# ------------------------
@app.route("/upload_seller", methods=["POST"])
def upload_seller():
    global stored_features, seller_image_hash

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Expected application/json"
        }), 415

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "status": "error",
            "message": "No image received"
        }), 400

    try:
        image_bytes = base64.b64decode(data["image"])
        seller_path = os.path.join(UPLOAD_FOLDER, "seller.jpg")

        with open(seller_path, "wb") as f:
            f.write(image_bytes)

        # Compute hash AFTER saving image
        seller_image_hash = image_hash(seller_path)

        # Extract features
        features = detect_cow_features(seller_path)

        if features is None:
            stored_features = None
            seller_image_hash = None
            return jsonify({
                "status": "error",
                "message": "Cow not detected clearly"
            }), 400

        stored_features = features

        print("✅ Seller features stored")

        return jsonify({
            "status": "success",
            "message": "Seller cow registered successfully"
        })

    except Exception as e:
        stored_features = None
        seller_image_hash = None
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ------------------------
# BUYER UPLOAD + MATCH
# ------------------------
@app.route("/upload_buyer", methods=["POST"])
def upload_buyer():
    global stored_features, seller_image_hash

    if stored_features is None or seller_image_hash is None:
        return jsonify({
            "status": "error",
            "message": "Upload seller cow first"
        }), 400

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Expected application/json"
        }), 415

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "status": "error",
            "message": "No image received"
        }), 400

    try:
        image_bytes = base64.b64decode(data["image"])
        buyer_path = os.path.join(UPLOAD_FOLDER, "buyer.jpg")

        with open(buyer_path, "wb") as f:
            f.write(image_bytes)

        buyer_hash = image_hash(buyer_path)

        # 🚫 SAME IMAGE PROTECTION
        if buyer_hash == seller_image_hash:
            return jsonify({
                "status": "error",
                "message": "Same image uploaded for buyer and seller"
            }), 400

        # Extract buyer features
        buyer_features = detect_cow_features(buyer_path)

        if buyer_features is None:
            return jsonify({
                "status": "error",
                "message": "Cow not detected clearly"
            }), 400

        # Similarity calculation
        similarity = calculate_similarity(stored_features, buyer_features)

        # 🛑 HARD SAFETY CLAMP
        if similarity < 0 or similarity > 1:
            similarity = 0.0

        percent = round(similarity * 100, 2)

        print(f"🧮 Match score: {percent}%")

        return jsonify({
            "status": "approved" if percent >= 80 else "rejected",
            "match_percent": percent,
            "message": (
                f"Match {percent}% – Approved"
                if percent >= 80
                else f"Match {percent}% – Not Approved"
            )
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ------------------------
# RUN SERVER
# ------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
