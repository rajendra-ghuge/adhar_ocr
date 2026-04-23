from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.inference import router
import os
import shutil

from contextlib import asynccontextmanager

def delete_contents(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Ensure static directory exists
TEMP_DIR = "app/static/output"
os.makedirs(TEMP_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Clean the folder before starting
    print("Startup: Cleaning output directory...")
    delete_contents(TEMP_DIR)
    yield
    # Shutdown: Clean the folder before stopping
    print("Shutdown: Cleaning output directory...")
    delete_contents(TEMP_DIR)

app = FastAPI(title="YOLO OCR API", lifespan=lifespan)

app.mount("/output", StaticFiles(directory=TEMP_DIR), name="output")
app.include_router(router)
