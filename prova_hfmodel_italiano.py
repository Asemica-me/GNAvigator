from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import requests

load_dotenv()
api_key = os.getenv("HF_TOKEN")
model_name = "google/gemma-3-27b-it" # modello italiano di Google
prompt = "Scrivi una breve storia su Roma in italiano."

# Use the raw API endpoint for text generation
API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
headers = {"Authorization": f"Bearer {api_key}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": prompt,
    "parameters": {"max_new_tokens": 200}
})

print(output[0]["generated_text"])