import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Custom Model
model = load_model("vegetable_classifier_model.h5")

# Load ImageNet Model (MobileNetV2) for filtering
imagenet_model = MobileNetV2(weights='imagenet')

classes = [
    'Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli',
    'Cabbage','Capsicum','Carrot','Cauliflower','Cucumber',
    'Papaya','Potato','Pumpkin','Radish','Tomato'
]

# Keywords to identify vegetables/fruits/food in ImageNet predictions
VEGETABLE_KEYWORDS = [
    'background', # rare case
    'vegetable', 'fruit', 'berry', 'melon', 'squash', 'mushroom', 'corn', 'root', 'tuber',
    'broccoli', 'cauliflower', 'zucchini', 'cucumber', 'pepper', 'cabbage', 'lettuce', 
    'spinach', 'pumpkin', 'radish', 'potato', 'tomato', 'bean', 'carrot', 'eggplant', 
    'gourd', 'papaya', 'artichoke', 'cardoon', 'sorrel', 'head_cabbage', 'rapeseed',
    'daisy', 'yellow_lady\'s_slipper', 'buckeye', 'coral_fungus', 'agaric', 'gyromitra',
    'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet_tissue', 
    'banana', 'apple', 'orange', 'lemon', 'fig', 'pineapple', 'pomegranate', 'custard_apple', 'jackfruit',
    'strawberry', 'potpie', 'burrito', 'pizza', 'carbonara', 'mashed_potato', 'guacamole', 'consomme',
    'trifle', 'ice_cream', 'ice_lolly', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'sandwich',
    'bell_pepper', 'chity', 'masala', 'boletus', 'hen_of_the_woods', 'tamarind', 'menu' 
]

def is_vegetable_candidate_array(img_array):
    """
    Checks if the input image array (preprocessed for MobileNetV2) 
    contains a vegetable candidate.
    """
    preds = imagenet_model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]
    # print("ImageNet Predictions:", decoded)
    
    for _, label, _ in decoded:
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in VEGETABLE_KEYWORDS):
            return True
    return False

def predict_single_image(img_path_or_array, is_path=True):
    """
    Runs the 2-stage prediction on a single image (path or array).
    Returns (ClassName, Confidence) or (None, None).
    """
    # Prepare image for ImageNet
    if is_path:
        img = image.load_img(img_path_or_array, target_size=(224,224))
        img_array = image.img_to_array(img)
    else:
        # Resize array to 224x224
        img_array = cv2.resize(img_path_or_array, (224, 224))
        # RGB format for Keras
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    img_array_inet = np.expand_dims(img_array, axis=0)
    img_array_inet_pre = preprocess_input(img_array_inet.copy()) # [-1, 1]

    if is_vegetable_candidate_array(img_array_inet_pre):
        # Custom Prediction
        img_array_custom = img_array_inet / 255.0 # [0, 1]
        pred = model.predict(img_array_custom)
        class_idx = np.argmax(pred)
        return classes[class_idx], np.max(pred)
    
    return None, None

def process_multiple_detection(image_path):
    """
    Detects objects, classifies them, and returns results + processed image path.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path, [], 0

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection and Dilation to close gaps
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = [] # List of {"class": "Tomato", "box": (x,y,w,h)}
    counts = {}
    
    height, width = img.shape[:2]
    min_area = (width * height) * 0.01 # Ignore objects smaller than 1% of image
    
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract ROI
        roi = original[y:y+h, x:x+w]
        
        # Predict
        label, conf = predict_single_image(roi, is_path=False)
        
        if label:
            detections.append({'class': label, 'box': [x, y, w, h]})
            counts[label] = counts.get(label, 0) + 1
            
            # Draw Box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Fallback: If no objects detected (or all filtered out), try whole image
    if not detections:
        label, conf = predict_single_image(image_path, is_path=True)
        if label:
            detections.append({'class': label, 'box': []})
            counts[label] = 1
            # Return original image as we didn't draw anything valid
            return image_path, detections, counts

    # Save processed image
    filename = "processed_" + os.path.basename(image_path)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(save_path, img)
    
    return save_path, detections, counts

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = []
    image_path = None
    counts = {}
    total_count = 0

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            # Simulate processing delay
            time.sleep(2.5)
            
            # Run Multi-Detection
            processed_path, detections, detected_counts = process_multiple_detection(image_path)
            
            # Update image path to point to processed image if detections occurred
            if detections and "processed_" in processed_path:
                image_path = processed_path
                
            prediction_results = detections
            counts = detected_counts
            total_count = sum(counts.values())

    return render_template(
        "index.html",
        image_path=image_path,
        predictions=prediction_results,
        counts=counts,
        total_count=total_count
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
