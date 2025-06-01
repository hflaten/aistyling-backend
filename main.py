from fastapi import FastAPI
from pydantic import BaseModel
import os
import replicate
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Setup Replicate client
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN is not set in environment variables")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Define request schema
class Prompt(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_outputs: Optional[int] = 1

# Define response schema
class ImageResponse(BaseModel):
    image_urls: List[str]

# Endpoint to generate image
@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        output = replicate_client.run(
            "stability-ai/stable-diffusion",
            input={
                "prompt": data.prompt,
                "width": data.width,
                "height": data.height,
                "num_outputs": data.num_outputs
            }
        )

        if isinstance(output, list):
            return {"image_urls": output}
        else:
            return {"image_urls": [output]}

    except Exception as e:
        return {"image_urls": [f"Error: {str(e)}"]}
