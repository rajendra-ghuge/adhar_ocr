import fitz  # PyMuPDF
import numpy as np
import cv2

def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        _, img_encoded = cv2.imencode(".jpg", img)
        images.append(img_encoded.tobytes())

    return images