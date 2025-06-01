from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Validate API token exists
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set. Please check your environment variables or .env file.")

# Define input schema
class Prompt(BaseModel):
    prompt: str

# Define output schema
class ImageResponse(BaseModel):
    image_urls: List[str]

# Image generation endpoint
@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        output = replicate.run(
            "stability-ai/stable-diffusion",
            input={"prompt": data.prompt},
            api_token=REPLICATE_API_TOKEN
        )

        if isinstance(output, list):
            return {"image_urls": output}
        else:
            return {"image_urls": [output]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
