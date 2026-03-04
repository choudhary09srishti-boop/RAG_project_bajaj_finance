from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get Hugging Face API key
hf_key = os.getenv("HUGGINGFACE_API_KEY")

if hf_key:
    print("Hugging Face API key loaded successfully!")
else:
    print("API key NOT found. Check your .env file.")
