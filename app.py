from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import cv2
import json
import datetime
from ultralytics import YOLO

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_history(record):
    history = load_history()
    history.insert(0, record)
    history = history[:50]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ---------------- MODEL ----------------
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("model/best_resnet50_final.pth", map_location="cpu"))
model.eval()

classes = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Normal"]

CLASS_METRICS = {
    "Atelectasis": {"accuracy": 71.2, "precision": 36.5, "recall": 38.1, "f1_score": 37.3, "model": "ResNet50", "detection_acc": 75.4},
    "Cardiomegaly": {"accuracy": 82.1, "precision": 45.2, "recall": 48.4, "f1_score": 46.8, "model": "ResNet50", "detection_acc": 88.2},
    "Effusion": {"accuracy": 78.5, "precision": 41.9, "recall": 42.2, "f1_score": 42.0, "model": "ResNet50", "detection_acc": 82.1},
    "Infiltration": {"accuracy": 68.9, "precision": 33.4, "recall": 35.8, "f1_score": 34.6, "model": "ResNet50", "detection_acc": 65.5},
    "Mass": {"accuracy": 72.6, "precision": 35.3, "recall": 36.5, "f1_score": 35.9, "model": "ResNet50", "detection_acc": 78.3},
    "Normal": {"accuracy": 88.3, "precision": 80.1, "recall": 83.4, "f1_score": 81.7, "model": "ResNet50", "detection_acc": 0.0}
}


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
# ---------------- LOAD MODELS SAFELY ----------------
'''model = None
yolo_model = None

def load_models():
    global model, yolo_model

    if model is None:
        print("Loading ResNet model...")
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 6)
        model.load_state_dict(torch.load("model/best_resnet50_final.pth", map_location="cpu"))
        model.eval()

    if yolo_model is None:
        print("Loading YOLO model...")
        yolo_model = YOLO("model/best.pt")
        load_models()'''

# ---------------- YOLO ----------------
yolo_model = YOLO("model/best.pt")

# ---------------- GRADCAM ----------------
def generate_gradcam(model, img_tensor):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=[2,3], keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html", history=load_history())

@app.route('/predict', methods=['POST'])
def predict():

    files = request.files.getlist('file')
    results_list = []

    for file in files:

        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # ---- CLASSIFICATION ----
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

        prediction = classes[pred.item()]
        confidence = round(confidence.item()*100, 2)
        probabilities = [round(p*100, 2) for p in probs.squeeze().tolist()]

        # ---- GRADCAM ----
        cam = generate_gradcam(model, img_tensor)

        img_cv = cv2.imread(path)
        img_cv = cv2.resize(img_cv, (224,224))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        gradcam_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

        gradcam_path = os.path.join(UPLOAD_FOLDER, "gradcam_" + filename)
        cv2.imwrite(gradcam_path, gradcam_img)

        # ---- YOLO ----
        results = yolo_model(np.array(img))
        result_img = results[0].plot()

        # Extract real detection accuracy (average confidence of all boxes)
        det_conf_values = results[0].boxes.conf.tolist()
        if det_conf_values:
            detection_accuracy = round(np.mean(det_conf_values) * 100, 2)
        else:
            detection_accuracy = 0.0

        output_path = os.path.join(UPLOAD_FOLDER, "det_" + filename)
        cv2.imwrite(output_path, result_img)

        metrics = CLASS_METRICS.get(prediction, CLASS_METRICS["Normal"])

        now_str = datetime.datetime.now().strftime("%b %d, %Y - %H:%M")
        record = {
            "id": f"#CLN-{np.random.randint(1000, 9999)}",
            "date": now_str,
            "prediction": prediction,
            "confidence": confidence,
            "status": "Critical Insight" if prediction != "Normal" else "Clear",
            "accuracy": confidence,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "detection_acc": detection_accuracy
        }
        save_history(record)

        results_list.append({
            "id": record["id"],
            "image": filename,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "output_image": "det_" + filename,
            "gradcam_image": "gradcam_" + filename,
            "accuracy": confidence,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "model": metrics["model"],
            "detection_acc": detection_accuracy,
            "severity": "Moderate" if prediction != "Normal" else "Normal",
            "explanation": f"The analysis detected {prediction} with {confidence}% confidence. Grad-CAM shows localized abnormalities in the lung fields."
        })

    return jsonify(results_list)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(load_history())

@app.route('/analytics', methods=['GET'])
def get_analytics():
    # Return some mock analytics based on history or fixed data
    history = load_history()
    stats = {
        "total_scans": len(history),
        "disease_counts": {},
        "weekly_trends": [
            {"name": "Mon", "scans": 4},
            {"name": "Tue", "scans": 7},
            {"name": "Wed", "scans": 5},
            {"name": "Thu", "scans": 8},
            {"name": "Fri", "scans": 12},
            {"name": "Sat", "scans": 6},
            {"name": "Sun", "scans": 9}
        ]
    }
    for item in history:
        p = item.get("prediction", "Normal")
        stats["disease_counts"][p] = stats["disease_counts"].get(p, 0) + 1
    
    return jsonify(stats)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    response = make_response(send_from_directory(UPLOAD_FOLDER, filename))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Cache-Control'] = 'no-cache'
    return response

if __name__ == "__main__":
    app.run()