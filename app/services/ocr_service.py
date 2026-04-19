import pytesseract
import cv2
import re

AADHAAR_REGEX = r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
DOB_REGEX = r"\b(\d{2}/\d{2}/\d{4}|\d{4})\b"
NAME_REGEX = r"^[A-Za-z\s]+$"
GENDER_REGEX = r"\b(MALE|FEMALE)\b"


def clean_text(text):
    # Remove common OCR artifacts (newlines, form feed characters)
    text = text.replace('\n', ' ').replace('\x0c', '')

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Trim leading/trailing whitespace
    text = text.strip()
    return text


def extract_text_from_crops(crops):
    result = {
        "aadhaar": None,
        "dob": None,
        "name": None,
        "gender": None,
        "address": None
    }

    roi_images = [] # Store ROI image crops, not OCR'd text yet

    for item in crops:
        crop = item["image"]
        class_id = item["class_id"]  # This is now a string label from process_detections

        # If it's an ROI crop, store the image and continue
        if class_id == "roi":
            roi_images.append(crop)
            continue

        # Process non-ROI crops first
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray, config="--psm 7")
        text = clean_text(text)  # Apply universal cleaning
        text_upper = text.upper()

        # Aadhaar
        if class_id == "adhar_no" and result["aadhaar"] is None:
            # Apply digit-focused corrections for Aadhaar
            cleaned_adhar_text = text
            cleaned_adhar_text = cleaned_adhar_text.replace('O', '0')
            cleaned_adhar_text = cleaned_adhar_text.replace('Q', '0')
            cleaned_adhar_text = cleaned_adhar_text.replace('l', '1')  # lowercase L
            cleaned_adhar_text = cleaned_adhar_text.replace('I', '1')  # uppercase I
            cleaned_adhar_text = cleaned_adhar_text.replace('L', '1')  # uppercase L
            cleaned_adhar_text = cleaned_adhar_text.replace('S', '5')
            cleaned_adhar_text = cleaned_adhar_text.replace('B', '8')
            cleaned_adhar_text = cleaned_adhar_text.replace('Z', '2')
            cleaned_adhar_text = cleaned_adhar_text.replace('A', '4')  # 'A' can look like '4'
            cleaned_adhar_text = cleaned_adhar_text.replace('G', '6')  # 'G' can look like '6'

            match = re.search(AADHAAR_REGEX, cleaned_adhar_text)
            if match:
                result["aadhaar"] = match.group()

        # DOB
        elif class_id == "dob" and result["dob"] is None:
            # Apply digit-focused corrections for DOB
            cleaned_dob_text = text
            cleaned_dob_text = cleaned_dob_text.replace('O', '0')
            cleaned_dob_text = cleaned_dob_text.replace('Q', '0')
            cleaned_dob_text = cleaned_dob_text.replace('l', '1')
            cleaned_dob_text = cleaned_dob_text.replace('I', '1')
            cleaned_dob_text = cleaned_dob_text.replace('L', '1')
            cleaned_dob_text = cleaned_dob_text.replace('S', '5')
            cleaned_dob_text = cleaned_dob_text.replace('B', '8')
            cleaned_dob_text = cleaned_dob_text.replace('Z', '2')
            cleaned_dob_text = cleaned_dob_text.replace('A', '4')
            cleaned_dob_text = cleaned_dob_text.replace('G', '6')

            match = re.search(DOB_REGEX, cleaned_dob_text)
            if match:
                result["dob"] = match.group()

        # Gender
        elif class_id == "gender" and result["gender"] is None:
            # For gender, use the original text after universal cleaning, case-insensitive match
            match = re.search(GENDER_REGEX, text_upper)
            if match:
                result["gender"] = match.group().capitalize()

        # Name
        elif class_id == "name" and result["name"] is None:
            # For name, we primarily want alphabetic characters and spaces.
            # The NAME_REGEX will handle filtering.
            if re.match(NAME_REGEX, text):  # text here is after universal clean_text
                # Avoid common labels
                if text.upper() not in ["NAME", "FATHER", "FATHER'S NAME", "ADDRESS"]:
                    result["name"] = text

        # Address
        elif class_id == "addr" and result["address"] is None:
            # Filter non-ASCII (remove Marathi/other regional scripts)
            english_only_text = re.sub(r'[^\x00-\x7F]+', '', text)
            # Clean up extra spaces that might remain after filtering
            english_only_text = re.sub(r'\s+', ' ', english_only_text).strip()
            
            if len(english_only_text) > 10: # Avoid tiny fragments
                result["address"] = english_only_text

    # 🔥 ROI fallback: Only process ROI images if any of the primary fields are still None
    if any(result[field] is None for field in ["aadhaar", "dob", "name", "gender"]):
        for roi_img in roi_images:
            # If all target fields are now filled, we can stop processing ROIs
            if all(result[field] is not None for field in ["aadhaar", "dob", "name", "gender"]):
                break

            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            text_from_roi = pytesseract.image_to_string(gray, config="--psm 6")
            text_from_roi = clean_text(text_from_roi)
            text_from_roi_upper = text_from_roi.upper()

            # Aadhaar fallback
            if result["aadhaar"] is None:
                cleaned_adhar_roi_text = text_from_roi
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('O', '0')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('Q', '0')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('l', '1')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('I', '1')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('L', '1')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('S', '5')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('B', '8')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('Z', '2')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('A', '4')
                cleaned_adhar_roi_text = cleaned_adhar_roi_text.replace('G', '6')

                match = re.search(AADHAAR_REGEX, cleaned_adhar_roi_text)
                if match:
                    result["aadhaar"] = match.group()

            # DOB fallback
            if result["dob"] is None:
                cleaned_dob_roi_text = text_from_roi
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('O', '0')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('Q', '0')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('l', '1')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('I', '1')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('L', '1')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('S', '5')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('B', '8')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('Z', '2')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('A', '4')
                cleaned_dob_roi_text = cleaned_dob_roi_text.replace('G', '6')

                match = re.search(DOB_REGEX, cleaned_dob_roi_text)
                if match:
                    result["dob"] = match.group()

            # Gender fallback
            if result["gender"] is None:
                match = re.search(GENDER_REGEX, text_from_roi_upper)  # text_from_roi_upper is already uppercase
                if match:
                    result["gender"] = match.group().capitalize()

            # Name fallback
            if result["name"] is None:
                clean_name = re.sub(r"[^A-Za-z\s]", "", text_from_roi)
                if re.match(NAME_REGEX, clean_name) and clean_name.strip():
                    result["name"] = clean_name.strip()

            # Address fallback
            if result["address"] is None:
                eng_addr_fallback = re.sub(r'[^\x00-\x7F]+', '', text_from_roi)
                eng_addr_fallback = re.sub(r'\s+', ' ', eng_addr_fallback).strip()
                if len(eng_addr_fallback) > 15:
                    result["address"] = eng_addr_fallback

    return result
