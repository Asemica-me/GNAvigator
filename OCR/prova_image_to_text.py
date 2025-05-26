# questo modello è abbastanza scadente per fare OCR di contenuto in italiano, ma è un buon esempio di come usare il modello italiano di Google per generare testo.
from dotenv import load_dotenv
import os
import requests
from PIL import Image
from transformers import pipeline

load_dotenv()
api_key = os.getenv("HF_TOKEN")

# Load image
image_url = "https://gna.cultura.gov.it/wiki/images/7/7c/VRD.jpg"
try:
    response_image = requests.get(image_url, stream=True)
    response_image.raise_for_status()  # Raise an exception for bad status codes
    image = Image.open(response_image.raw).convert("RGB")
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# Use the pipeline
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
caption = image_to_text(image)[0]["generated_text"]
print("Image Caption:", caption)