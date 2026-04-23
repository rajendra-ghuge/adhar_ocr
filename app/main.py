from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.inference import router
import os
import shutil

app = FastAPI(title="YOLO OCR API")

# Ensure static directory exists
TEMP_DIR = "app/static/output"
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/output", StaticFiles(directory=TEMP_DIR), name="output")
app.include_router(router)

@app.on_event("shutdown")
def cleanup_temp_files():
    if os.path.exists(TEMP_DIR):
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
