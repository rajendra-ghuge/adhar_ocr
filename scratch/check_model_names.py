from ultralytics import YOLO
import os

model_path = r"c:\Users\Rajendra\Desktop\Yolo_ocr\project-root\app\models\best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print("Model Classes:", model.names)
else:
    print(f"Model not found at {model_path}")
