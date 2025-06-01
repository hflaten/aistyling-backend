from fastapi import FastAPI
from pydantic import BaseModel
import os
import replicate
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Setup Replicate client
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set in environment variables")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Define request schema
class Prompt(BaseModel):
    prompt: str

# Define response schema
class ImageResponse(BaseModel):
    image_urls: List[str]

# Endpoint to generate image
@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        output = client.run(
            "stability-ai/stable-diffusion@b3d14e1cd1f9470bbb0bb68cac48e5f483e5be309551992cc33dc30654a82bb7",
            input={"prompt": data.prompt}
        )

        if isinstance(output, list):
            return {"image_urls": output}
        else:
            return {"image_urls": [output]}

    except Exception as e:
        return {"image_urls": [f"Error: {str(e)}"]}
