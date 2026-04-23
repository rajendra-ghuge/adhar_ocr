from fastapi import APIRouter, UploadFile, File
from app.services.yolo_service import detect_objects, save_visualized_image
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
        
        # Save visualized image
        viz_filename = save_visualized_image(img_bytes, detections)
        viz_url = f"/output/{viz_filename}" if viz_filename else None

        results.append({
            "detections": detections,
            "texts": texts,
            "visualized_image": viz_url
        })

    # Return results from the first image (consistent with original logic)
    # but include the visualized image URL
    return {
        "results": results[0]["texts"],
        "visualized_image": results[0]["visualized_image"]
    }