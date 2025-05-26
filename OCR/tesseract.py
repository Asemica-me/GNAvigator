import pytesseract
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def preprocess_image_v2(img, scale_factor=3.0):
#     """Enhanced preprocessing for potentially better OCR"""
#     cv_img = np.array(img.convert('L')) # Convert to grayscale directly

#     # Scale up
#     scaled = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

#     # Apply adaptive histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(scaled)

#     return Image.fromarray(equalized)

# def preprocess_image_v4(img, scale_factor=4.0):
#     """More aggressive scaling and contrast enhancement"""
#     cv_img = np.array(img.convert('L'))

#     # Scaling
#     scaled = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

#     # Increase contrast
#     alpha = 1.2  # Contrast control (1.0-3.0)
#     beta = 0    # Brightness control (0-100)
#     contrast_enhanced = cv2.convertScaleAbs(scaled, alpha=alpha, beta=beta)

#     # Apply adaptive histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(contrast_enhanced)

#     return Image.fromarray(equalized)

# def preprocess_image_v5(img, scale_factor=4.0):
#     """More contrast and scaling"""
#     cv_img = np.array(img.convert('L'))

#     # Scaling
#     scaled = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

#     # Apply a stronger contrast stretching
#     minVal, maxVal, _, _ = cv2.minMaxLoc(scaled)
#     stretched = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#     # Apply adaptive histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     equalized = clahe.apply(stretched)

#     return Image.fromarray(equalized)

def preprocess_image_general(img, scale_factor=2.5):
    """General preprocessing for varied images"""
    cv_img = np.array(img.convert('L'))

    # Moderate scaling
    scaled = cv2.resize(cv_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(scaled)

    return Image.fromarray(equalized)

# Download and process image
image_url = "https://gna.cultura.gov.it/wiki/images/4/49/Infarinato_04jpg.jpg"
response = requests.get(image_url)
try:
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    exit()
except Image.UnidentifiedImageError:
    print("Invalid image format")
    exit()

# Apply the new preprocessing
processed_img = preprocess_image_general(img, scale_factor=4.0)

custom_config = r'''
    --oem 3
    --psm 3
    -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzàèéìòùç°§0123456789,;:!?()@\@{}+=_[]/\\- 
    -c preserve_interword_spaces=1
'''
text = pytesseract.image_to_string(
    processed_img,
    config=custom_config,
    lang='ita'
)

# # Display comparison
# plt.figure(figsize=(15, 8))

# # Original image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(img)
# plt.axis('off')

# # Preprocessed image
# plt.subplot(1, 2, 2)
# plt.title("Preprocessed Image")
# plt.imshow(processed_img, cmap='gray')
# plt.axis('off')

# plt.show()

print("\nOCR Output (after preprocessing):")
raw_text_after_preprocess = pytesseract.image_to_string(processed_img, lang='ita', config=custom_config)
print(raw_text_after_preprocess)