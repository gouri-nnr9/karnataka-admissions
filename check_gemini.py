from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai

# Load .env from this project's folder.
load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY was not found in your .env file.")

print("API key loaded successfully.")

# Check the connection by listing model names.
with genai.Client(api_key=api_key) as client:
    for model in client.models.list():
        if "gemini" in (model.name or ""):
            print(model.name)