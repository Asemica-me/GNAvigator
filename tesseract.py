import pytesseract
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img, scale_factor=2.5):
    """General preprocessing for varied images"""
    cv_img = np.array(img.convert('L'))
    scaled = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(scaled)
    return Image.fromarray(equalized)

def extract_text_from_image(image_url, scale_factor=4.0, custom_config=None, lang='ita'):
    """
    Extracts text from an image URL using OCR.
    
    Args:
        image_url (str): URL of the image to process.
        scale_factor (float): Scaling factor for image preprocessing (default: 4.0).
        custom_config (str): Custom Tesseract OCR configuration (default: None).
        lang (str): Language for OCR (default: 'ita').
    
    Returns:
        str: Extracted text, or None if an error occurs.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    
    try:
        img = Image.open(BytesIO(response.content))
    except Image.UnidentifiedImageError:
        print("Invalid image format")
        return None
    
    processed_img = preprocess_image(img, scale_factor=scale_factor)
    
    if custom_config is None:
        custom_config = r'''
            --oem 3
            --psm 3
            -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzàèéìòùç°§0123456789,;:!?()@\@{}+=_[]/\\- 
            -c preserve_interword_spaces=1
        '''
    
    try:
        text = pytesseract.image_to_string(processed_img, config=custom_config, lang=lang)
    except pytesseract.TesseractError as e:
        print(f"OCR Error: {e}")
        return None
    
    return text

if __name__ == "__main__":
    # Example usage
    image_url = "https://gna.cultura.gov.it/wiki/images/4/49/Infarinato_04jpg.jpg"
    extracted_text = extract_text_from_image(image_url)
    
    if extracted_text:
        print("\nOCR Output:")
        print(extracted_text)