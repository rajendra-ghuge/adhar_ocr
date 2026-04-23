from ultralytics import YOLO
import numpy as np
import cv2
import pytesseract
import re
import uuid
import os
from app.services.ocr_service import TESSERACT_CONFIG

model = YOLO("app/models/best.pt")

def detect_objects(img):
    # If input is bytes (e.g., from FastAPI UploadFile), decode it
    if isinstance(img, bytes):
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print("Error: Input image is None or could not be decoded.")
        return [], []

    results = model.predict(
        img,
        conf=0.30)
    
    print(f"[YOLO Log] Detected {sum(len(r.boxes) for r in results)} boxes total.")
    
    detections = []
    crops = []
    name_candidates = []
    address_candidates = []

    # Ensure img is not None before proceeding
    if img is None:
        print("Error: Input image is None.")
        return [], []

    h, w = img.shape[:2]

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id_int = int(box.cls[0])
            confidence = float(box.conf[0])

            # Get the class label from the model's names attribute
            class_label = r.names[class_id_int]

            # ✅ Safe crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if not (x2 > x1 and y2 > y1):
                print(f"[YOLO Log] Invalid box for {class_label}: [{x1}, {y1}, {x2}, {y2}]")
                continue

            current_class_id = class_label
            # No longer skipping addr as per user request

            crop = img[y1:y2, x1:x2]

            # If it's a 'name' or 'addr' class, collect it for filtering
            if current_class_id == 'name':
                name_candidates.append({
                    "image": crop,
                    "class_id": current_class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
            elif current_class_id == 'addr':
                address_candidates.append({
                    "image": crop,
                    "class_id": current_class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
            else:
                crops.append({
                    "image": crop,
                    "class_id": current_class_id,
                    "confidence": confidence
                })
                detections.append({
                    "class_id": current_class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

    # 🔥 Filter 'name' candidates to keep only the English name
    if name_candidates:
        best_name_crop = None
        max_score = -1

        for candidate in name_candidates:
            # Quick OCR check
            gray_crop = cv2.cvtColor(candidate["image"], cv2.COLOR_BGR2GRAY)
            h_c, w_c = gray_crop.shape
            config = TESSERACT_CONFIG.get("name", "--oem 3 --psm 7")
            text = pytesseract.image_to_string(gray_crop, config=config).strip()
            print(f"[OCR Candidate Log] Class: name | Size: {w_c}x{h_c} | Config: {config} | Raw Result: '{text}'")
            
            # Clean text: keep only alphanumeric
            clean_text = re.sub(r'[^A-Za-z]', '', text)
            
            # Score based on:
            # 1. Does it NOT contain the word "NAME"?
            # 2. Length of English characters
            lower_text = text.lower()
            if "name" in lower_text or "नाम" in text:
                score = 0
            else:
                score = len(clean_text)

            if score > max_score:
                max_score = score
                best_name_crop = candidate

        if best_name_crop:
            print(f"[YOLO Log] Selected best 'name' crop with confidence {best_name_crop['confidence']}")
            crops.append({
                "image": best_name_crop["image"],
                "class_id": best_name_crop["class_id"],
                "confidence": best_name_crop["confidence"]
            })
            detections.append({
                "class_id": best_name_crop["class_id"],
                "confidence": best_name_crop["confidence"],
                "bbox": best_name_crop["bbox"]
            })

    # 🔥 Filter 'addr' candidates to keep only the English address
    if address_candidates:
        best_addr_crop = None
        max_addr_score = -1

        for candidate in address_candidates:
            # Quick OCR check
            gray_crop = cv2.cvtColor(candidate["image"], cv2.COLOR_BGR2GRAY)
            h_c, w_c = gray_crop.shape
            config = TESSERACT_CONFIG.get("addr", "--oem 3 --psm 6")
            text = pytesseract.image_to_string(gray_crop, config=config).strip()
            print(f"[OCR Candidate Log] Class: addr | Size: {w_c}x{h_c} | Config: {config} | Raw Result: '{text}'")
            
            # Clean text: keep only ASCII alphanumeric and common address symbols
            clean_text = re.sub(r'[^A-Za-z0-9#\-/.,\s]', '', text)
            
            # Score based on length of ASCII characters
            score = len(clean_text)

            if score > max_addr_score:
                max_addr_score = score
                best_addr_crop = candidate

        if best_addr_crop:
            print(f"[YOLO Log] Selected best 'addr' crop with confidence {best_addr_crop['confidence']}")
            crops.append({
                "image": best_addr_crop["image"],
                "class_id": best_addr_crop["class_id"],
                "confidence": best_addr_crop["confidence"]
            })
            detections.append({
                "class_id": best_addr_crop["class_id"],
                "confidence": best_addr_crop["confidence"],
                "bbox": best_addr_crop["bbox"]
            })

    return detections, crops

def visualize_detections(img, detections):
    # Colors for the 6 classes (BGR)
    color_map = {
        'addr': (255, 0, 0),        # Blue
        'adhar_no': (0, 255, 0),    # Green
        'dob': (0, 0, 255),         # Red
        'gender': (255, 255, 0),    # Cyan
        'name': (255, 0, 255),      # Magenta
        'roi': (0, 255, 255)        # Yellow
    }
    
    # Ensure img is a numpy array
    if isinstance(img, bytes):
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    viz_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = color_map.get(det['class_id'], (128, 128, 128))
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
        
    h, w = viz_img.shape[:2]
    legend_w = 300
    legend = np.zeros((h, legend_w, 3), dtype=np.uint8) + 255
    
    cv2.putText(legend, "Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    y_offset = 70
    for det in detections:
        color = color_map.get(det['class_id'], (128, 128, 128))
        cv2.rectangle(legend, (10, y_offset - 20), (40, y_offset), color, -1)
        text = f"{det['class_id']}: {det['confidence']:.2f}"
        cv2.putText(legend, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 30
        if y_offset > h - 20: 
            break
        
    combined = np.hstack((viz_img, legend))
    return combined

def save_visualized_image(img, detections, output_dir="app/static/output"):
    combined = visualize_detections(img, detections)
    if combined is None:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"result_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, combined)
    return filename