from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import cv2
import numpy as np
import requests
import time
from datetime import datetime

# Configuration
MODEL_PATH = "../scripts/final_model"
FASTAPI_URL = "http://localhost:8000/api/waste"
MIN_CONFIDENCE = 0.9
DEBOUNCE_TIME = 2.0
ORGANIC_PAUSE_TIME = 3.0 
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Track last sent time for each class
last_sent_times = {"organic": 0, "non_organic": 0}

def classify_waste(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_class = torch.max(probs, dim=1)
    return {
        "class": model.config.id2label[top_class.item()],
        "confidence": top_prob.item()
    }

def send_to_api(class_name: str):
    current_time = time.time()
    if current_time - last_sent_times[class_name] > DEBOUNCE_TIME:
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"class_name": class_name},
                timeout=1.0
            )
            if response.status_code == 200:
                last_sent_times[class_name] = current_time
                return True
        except requests.RequestException as e:
            print(f"API request failed: {e}")
    return False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.resize(frame, (224, 224))
            result = classify_waste(frame)

            if result["confidence"] >= MIN_CONFIDENCE:
                class_name = "organic" if result["class"] == "organic" else "non_organic"
                if send_to_api(class_name):
                    if class_name == "organic":
                        print("Organic detected. Pausing for a moment...")
                        time.sleep(ORGANIC_PAUSE_TIME)

            label = f"{result['class']} ({result['confidence']:.2%})"


            if result["class"] == "LABEL_0":
                label = f"Organic {result['confidence']:.2%}"
                color = (0, 255, 0)  # Green for organic
            else:
                label = f"Non Organic {result['confidence']:.2%}"
                color = (0, 0, 255)  # Red for non-organic

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow('Waste Classifier', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
