from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import replicate
import os

app = FastAPI()

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
MODEL_REF = "fofr/swapstyle"

if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set")
if not FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN is not set")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageResponse(BaseModel):
    image_urls: List[str]

@app.post("/generate", response_model=ImageResponse)
async def generate_image(
    prompt: str = Form(...),
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    try:
        # Lagre midlertidige bilder
        user_image_bytes = await user_image.read()
        product_image_bytes = await product_image.read()

        # Last opp bildene til Replicate
        user_image_url = replicate.files.upload(user_image_bytes)
        product_image_url = replicate.files.upload(product_image_bytes)

        output = replicate.run(
            MODEL_REF,
            input={
                "prompt": prompt,
                "input_image_1": user_image_url,
                "input_image_2": product_image_url,
            },
        )

        # Returner bilde-URL-er
        urls = [str(item) for item in output] if isinstance(output, list) else [str(output)]
        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
