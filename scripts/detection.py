from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import cv2
import numpy as np
import requests
import time
from datetime import datetime
from threading import Thread
import queue

# Configuration
MODEL_PATH = "../scripts/final_model"
FASTAPI_URL = "http://localhost:8000/api/organic"
MIN_CONFIDENCE = 0.9
DEBOUNCE_TIME = 2.0
ORGANIC_PAUSE_TIME = 3.0

# Initialize model and processor
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Optimize model
model = model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("Using CUDA acceleration")
else:
    print("Using CPU")

# Track last sent time for each class
last_sent_times = {"organic": 0, "non_organic": 0}

# Queues for asynchronous processing
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
api_queue = queue.Queue()

def classify_waste(frame):
    # Convert frame to tensor
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    
    # Process with model
    inputs = processor(images=pil_image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get results
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_prob, top_class = torch.max(probs, dim=1)
    
    return {
        "class": model.config.id2label[top_class.item()],
        "confidence": top_prob.item()
    }

def inference_worker():
    while True:
        frame = frame_queue.get()
        if frame is None:  # Exit signal
            break
        try:
            result = classify_waste(frame)
            result_queue.put(result)
        except Exception as e:
            print(f"Inference error: {e}")

def api_worker():
    while True:
        data = api_queue.get()
        if data is None:  # Exit signal
            break
        
        current_time = time.time()
        class_name = data["class_name"]
        confidence = data["confidence"]
        print("-------------------------------")
        print(class_name)
        print("-------------------------------")
        # Only process organic detections
        if class_name == "organic":
            if current_time - last_sent_times["organic"] > DEBOUNCE_TIME:
                
                try:
                    params = {"confidence": float(confidence)}
                    response = requests.post(
                        FASTAPI_URL,
                        params=params,
                        timeout=1.0
                    )
                    
                    # Debug prints
                    print("-------------------------------")
                    print(f"Response Status: {response.status_code}")
                    print(f"Response Content: {response.text}")
                    print("-------------------------------")
                    if response.status_code == 200:
                        last_sent_times["organic"] = current_time
                        print(f"Successfully sent organic to API")
                except requests.RequestException as e:
                    print(f"API request failed: {e}")
        else:
            # Just log non-organic detections without API call
            print(f"Non-organic detected - not sending to API")


def main():
    infer_thread = Thread(target=inference_worker)
    api_thread = Thread(target=api_worker)
    infer_thread.start()
    api_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        last_result = None
        last_update_time = time.time()
        pause_until = 0
        is_paused = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            current_time = time.time()
            display_frame = cv2.resize(frame.copy(), (640, 480))

            # PAUSE MODE - No processing at all
            if is_paused:
                if current_time >= pause_until:
                    is_paused = False
                    print("Resuming normal operation")
                else:
                    # Clear any pending work
                    while not frame_queue.empty():
                        frame_queue.get_nowait()
                    while not result_queue.empty():
                        result_queue.get_nowait()

                    cv2.putText(display_frame, "PAUSED (Organic Detected)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow('Waste Classifier', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # NORMAL OPERATION
            if frame_queue.empty():
                try:
                    frame_queue.put_nowait(cv2.resize(frame, (224, 224)))
                except queue.Full:
                    pass

            try:
                result = result_queue.get_nowait()
                last_result = result
                last_update_time = current_time

                class_name = result["class"]
                confidence = result["confidence"]

                print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

                if class_name == "LABEL_0" and confidence >= MIN_CONFIDENCE:
                    api_queue.put({
            "class_name": "organic",
            "confidence": confidence  # Send the actual confidence value
        })
                    #api_queue.put("organic")
                    pause_until = time.time() + 7
                    is_paused = True
                    print(f"Pausing ALL predictions for 7 seconds (until {pause_until})")
                    continue

                elif confidence >= MIN_CONFIDENCE and class_name != "LABEL_0":
                    api_queue.put({
            "class_name": "non_organic",
            "confidence": confidence
        })

            except queue.Empty:
                pass

            # Display last result if recent
            if last_result and (time.time() - last_update_time < 1.0):
                label_text = "organic" if last_result["class"] == "LABEL_0" else "non organic"
                label = f"{label_text} ({last_result['confidence']:.2%})"
                color = (0, 255, 0) if last_result["class"] == "LABEL_0" else (0, 0, 255)
                cv2.putText(display_frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Show FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Waste Classifier', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        frame_queue.put(None)
        api_queue.put(None)
        infer_thread.join()
        api_thread.join()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()