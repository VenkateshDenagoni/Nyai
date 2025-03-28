import torch
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load API key from environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Define system prompt
from datetime import datetime, timedelta

today = datetime.today().strftime("%Y-%m-%d")
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SYSTEM_PROMPT = f"""You are Mistral Small 3.1, an AI model...
The current date is {today}. Yesterday was {yesterday}.
"""

def build_mistral_prompt(user_prompt: str) -> str:
    return f"<s>[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]"

# Initialize text-generation pipeline
generation_pipeline = pipeline(
    "text-generation",
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    tokenizer="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    trust_remote_code=True,
    device=0 if torch.cuda.is_available() else -1,
)

def generate_response(prompt: str):
    """Generates AI response using Mistral model."""
    final_prompt = build_mistral_prompt(prompt)
    output = generation_pipeline(
        final_prompt, max_length=100, do_sample=True, truncation=True, temperature=0.15
    )
    return output[0]["generated_text"] if output else ""
