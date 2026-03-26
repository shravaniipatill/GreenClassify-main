import os
import cv2
import numpy as np
import urllib.request
from app import process_multiple_detection, predict_single_image

def download_image(url, filename):
    try:
        print(f"Downloading {filename}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print("Download successful.")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def create_multi_image(img1_path, img2_path, output_path):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Cannot create multi-image: Input images missing.")
        return False
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resize to same height
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    
    # Concatenate horizontally
    combined = np.hstack((img1, img2))
    cv2.imwrite(output_path, combined)
    print(f"Created {output_path}")
    return True

def log(message):
    print(message)
    with open("verification_log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")

if __name__ == "__main__":
    # Clear log file
    with open("verification_log.txt", "w", encoding="utf-8") as f:
        f.write("Starting Verification...\n")

    # URLs
    tomato_url = "https://upload.wikimedia.org/wikipedia/commons/2/23/Tomato_je.jpg"
    potato_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Patates.jpg"
    dog_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

    download_image(tomato_url, "test_tomato.jpg")
    download_image(potato_url, "test_potato.jpg")
    download_image(dog_url, "test_dog.jpg")

    # 1. Test Multi-Detection (Stitched)
    log("\n--- Testing Multi-Detection (Tomato + Potato) ---")
    if create_multi_image("test_tomato.jpg", "test_potato.jpg", "test_multi.jpg"):
        processed_path, detections, counts = process_multiple_detection("test_multi.jpg")
        log(f"Counts: {counts}")
        log(f"Detections: {len(detections)}")
        
        if len(detections) >= 2:
            log("SUCCESS: Multiple objects detected.")
        else:
            log("WARNING: Less than 2 objects detected. Single object fallback might have triggered or segmentation failed.")
    
    # 2. Test Single Vegetable (Fallback/Direct)
    log("\n--- Testing Single Vegetable (Tomato) ---")
    if os.path.exists("test_tomato.jpg"):
        processed_path, detections, counts = process_multiple_detection("test_tomato.jpg")
        log(f"Counts: {counts}")
        if "Tomato" in counts:
            log("SUCCESS: Tomato detected.")
        else:
             log(f"FAIL: Expected Tomato, got {counts}")

    # 3. Test Non-Vegetable (Dog)
    log("\n--- Testing Non-Vegetable (Dog) ---")
    if os.path.exists("test_dog.jpg"):
        processed_path, detections, counts = process_multiple_detection("test_dog.jpg")
        log(f"Counts: {counts}")
        if not counts:
             log("SUCCESS: Dog rejected (No vegetables found).")
        else:
             log(f"FAIL: Dog misclassified as {counts}")
