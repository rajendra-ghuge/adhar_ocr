from fastapi import APIRouter, UploadFile, File
from app.services.yolo_service import detect_objects
from app.services.ocr_service import extract_text_from_crops
from app.services.pdf_service import pdf_to_images

router = APIRouter(prefix="/api")

@router.get("/")
def health():
    return {"status": "API running"}

@router.post("/process")
async def process_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        images = pdf_to_images(content)
    else:
        images = [content]

    results = []

    for img_bytes in images:
        detections, crops = detect_objects(img_bytes)
        texts = extract_text_from_crops(crops)

        results.append({
            "detections": detections,
            "texts": texts
        })

    return {"results": results[0]["texts"]}