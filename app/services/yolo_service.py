from ultralytics import YOLO
import numpy as np
import cv2
import pytesseract
import re

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
        conf=0.50,
        save=True
    )
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
            text = pytesseract.image_to_string(gray_crop, config="--psm 7").strip()
            
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
            text = pytesseract.image_to_string(gray_crop, config="--psm 6").strip()
            
            # Clean text: keep only ASCII alphanumeric and common address symbols
            clean_text = re.sub(r'[^A-Za-z0-9#\-/.,\s]', '', text)
            
            # Score based on length of ASCII characters
            score = len(clean_text)

            if score > max_addr_score:
                max_addr_score = score
                best_addr_crop = candidate

        if best_addr_crop:
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