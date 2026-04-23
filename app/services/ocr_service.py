import pytesseract
import cv2
import re

AADHAAR_REGEX = r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
DOB_REGEX = r"\b(\d{2}/\d{2}/\d{4}|(?:19|20)\d{2})\b"
NAME_REGEX = r"^[A-Za-z\s]+$"
GENDER_REGEX = r"\b(MALE|FEMALE)\b"

TESSERACT_CONFIG = {
    "name": "--oem 3 --psm 7 ",
    "dob": "--oem 3 --psm 7 ",
    "adhar_no": "--oem 3 --psm 7 ",
    "gender": "--oem 3 --psm 8 ",
    "addr": "--oem 3 --psm 6",
    "roi": "--oem 3 --psm 11"
}

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
        
        # Upscale for better space detection (especially for names)
        if class_id == "name":
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
        h_c, w_c = gray.shape
        
        config = TESSERACT_CONFIG.get(class_id, "--oem 3 --psm 7")
        text = pytesseract.image_to_string(gray, config=config)
        raw_text = text.strip()
        cleaned_text = text # Before specific regex
        print(f"[OCR Raw] Class: {class_id} | Size: {w_c}x{h_c} | Raw: '{raw_text}'")
        
        text = clean_text(text)
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
                print(f"[OCR Final] Field: aadhaar | Value: '{result['aadhaar']}'")

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

            # For DOB, prioritize full date matches (DD/MM/YYYY) over 4-digit years
            matches = re.findall(DOB_REGEX, cleaned_dob_text)
            if matches:
                # Find the first full date (contain '/') if it exists, otherwise use the first year
                full_dates = [m for m in matches if '/' in m]
                result["dob"] = full_dates[0] if full_dates else matches[0]
                print(f"[OCR Final] Field: dob | Value: '{result['dob']}'")

        # Gender
        elif class_id == "gender" and result["gender"] is None:
            # For gender, use the original text after universal cleaning, case-insensitive match
            match = re.search(GENDER_REGEX, text_upper)
            if match:
                result["gender"] = match.group().capitalize()
                print(f"[OCR Final] Field: gender | Value: '{result['gender']}'")

        # Name
        elif class_id == "name" and result["name"] is None:
            # For name, we primarily want alphabetic characters and spaces.
            # The NAME_REGEX will handle filtering.
            if re.match(NAME_REGEX, text):  # text here is after universal clean_text
                # Avoid common labels
                if text.upper() not in ["NAME", "FATHER", "FATHER'S NAME", "ADDRESS"]:
                    result["name"] = text
                    print(f"[OCR Final] Field: name | Value: '{result['name']}'")

        # Address
        elif class_id == "addr" and result["address"] is None:
            # Filter non-ASCII (remove Marathi/other regional scripts)
            english_only_text = re.sub(r'[^\x00-\x7F]+', '', text)
            # Clean up extra spaces that might remain after filtering
            english_only_text = re.sub(r'\s+', ' ', english_only_text).strip()
            
            # Remove "Address:" or "Add:" prefixes (case-insensitive)
            addr_prefix_regex = r"^(ADDRESS|ADDR|ADD)[:\s-]+"
            english_only_text = re.sub(addr_prefix_regex, "", english_only_text, flags=re.IGNORECASE).strip()
            
            if len(english_only_text) > 10: # Avoid tiny fragments
                result["address"] = english_only_text
                print(f"[OCR Final] Field: address | Value: '{result['address']}'")

    # 🔥 ROI fallback: Only process ROI images if any of the primary fields are still None
    target_fields = ["aadhaar", "dob", "gender", "name", "address"]
    if any(result[field] is None for field in target_fields):
        for roi_img in roi_images:
            # If all target fields are now filled, we can stop processing ROIs
            if all(result[field] is not None for field in target_fields):
                break

            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            h_c, w_c = gray.shape
            config = TESSERACT_CONFIG.get("roi", "--oem 3 --psm 11")
            text_from_roi = pytesseract.image_to_string(gray, config=config)
            raw_roi = text_from_roi.strip()
            print(f"[OCR Raw] Class: ROI | Size: {w_c}x{h_c} | Raw: '{raw_roi}'")
            
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
                    print(f"[OCR ROI Final] Field: aadhaar | Value: '{result['aadhaar']}'")

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

                # ROI logic: prioritize full dates
                matches = re.findall(DOB_REGEX, cleaned_dob_roi_text)
                if matches:
                    full_dates = [m for m in matches if '/' in m]
                    result["dob"] = full_dates[0] if full_dates else matches[0]
                    print(f"[OCR ROI Final] Field: dob | Value: '{result['dob']}'")

            # Gender fallback
            if result["gender"] is None:
                match = re.search(GENDER_REGEX, text_from_roi_upper)  # text_from_roi_upper is already uppercase
                if match:
                    result["gender"] = match.group().capitalize()
                    print(f"[OCR ROI Final] Field: gender | Value: '{result['gender']}'")

            # Name fallback
            

            # Removed Address fallback from ROI to prevent incorrect data capture
            pass

    return result
