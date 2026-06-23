from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
import uuid

# Load colorization model
color_net = cv2.dnn.readNetFromCaffe(
    "model/colorization_deploy_v2.prototxt",
    "model/colorization_release_v2.caffemodel"
)

pts = np.load("model/pts_in_hull.npy")

class8 = color_net.getLayerId("class8_ab")
conv8 = color_net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2,313,1,1)
color_net.getLayer(class8).blobs = [pts.astype(np.float32)]
color_net.getLayer(conv8).blobs = [np.full([1,313], 2.606, np.float32)]

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

data_store = {}

def colorize_image(img):
    img = img.astype(np.float32) / 255.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]

    L_resized = cv2.resize(L, (224,224))
    L_resized -= 50

    color_net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = color_net.forward()[0,:,:,:].transpose((1,2,0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = L[:,:,np.newaxis]
    lab_output = np.concatenate((L, ab), axis=2)

    bgr_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
    bgr_output = np.clip(bgr_output, 0, 1)

    return (bgr_output * 255).astype(np.uint8)

@app.route("/")
def index():
    return render_template("index.html")

# Upload + detect
@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    image = cv2.imread(path)

    image = colorize_image(image)

    results = model(image)[0]

    image_id = str(uuid.uuid4())

    data_store[image_id] = {
        "image": image.copy(),
        "results": results
    }

    boxes = []

    if results.boxes is not None:
        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            boxes.append({
                "id": i,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            })

            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

    out_path = os.path.join(OUTPUT_FOLDER, f"{image_id}_detected.jpg")
    cv2.imwrite(out_path, image)

    return jsonify({
    "image": out_path,
    "boxes": boxes,
    "image_id": image_id
    })

# Apply color
@app.route("/color", methods=["POST"])
def color():

    data = request.json
    obj_id = data["id"]
    color = data["color"]  # [R,G,B]

    image_id = data["image_id"]

    stored = data_store[image_id]
    image = stored["image"].copy()
    results = stored["results"]

    mask = results.masks.data[obj_id].cpu().numpy()

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 0.5).astype(np.uint8)

    color_layer = np.zeros_like(image)
    color_layer[:] = color[::-1]  # RGB → BGR

    # Blend color only on mask
# Convert to LAB for shading preservation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract lightness (this is shading!)
    L = lab[:,:,0]

    # Normalize lightness to [0,1]
    L_norm = L / 255.0

    # Create new color using shading
    b, g, r = color[::-1]

    new_b = (L_norm * b).astype(np.uint8)
    new_g = (L_norm * g).astype(np.uint8)
    new_r = (L_norm * r).astype(np.uint8)

    alpha = 0.8

    image[:,:,0] = np.where(mask==1, new_b*alpha + image[:,:,0]*(1-alpha), image[:,:,0])
    image[:,:,1] = np.where(mask==1, new_g*alpha + image[:,:,1]*(1-alpha), image[:,:,1])
    image[:,:,2] = np.where(mask==1, new_r*alpha + image[:,:,2]*(1-alpha), image[:,:,2])

    out_path = os.path.join(OUTPUT_FOLDER, f"{image_id}_colored.jpg")
    cv2.imwrite(out_path, image)

    return jsonify({"image": out_path})

if __name__ == "__main__":
    app.run(debug=True)
