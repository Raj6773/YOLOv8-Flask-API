import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

@app.route("/")
def home():
    return "🚀 YOLOv8 Flask API is running!"

# 📸 Image Upload for Detection
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img, imgsz=320)  # Reduce input image size to 320x320

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({"class": model.names[cls], "confidence": conf, "box": [x1, y1, x2, y2]})

    return jsonify({"detections": detections})

# 🎥 Real-Time Camera Detection
@app.route("/camera")
def camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[cls]} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Camera Detection Stopped"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
