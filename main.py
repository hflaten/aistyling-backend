import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List, Optional
import cloudinary.uploader
import replicate

# Last inn .env-verdier
load_dotenv()

# Konfigurer Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# FastAPI-app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modell
MODEL = "flux-kontext-apps/multi-image-kontext-max"

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    user_image: Optional[UploadFile] = None,
    product_image: Optional[UploadFile] = None
):
    try:
        if not user_image or not product_image:
            raise HTTPException(status_code=400, detail="Both user_image and product_image are required.")

        # Last opp til Cloudinary
        user_image_upload = cloudinary.uploader.upload(user_image.file)
        product_image_upload = cloudinary.uploader.upload(product_image.file)

        user_image_url = user_image_upload.get("secure_url")
        product_image_url = product_image_upload.get("secure_url")

        if not user_image_url or not product_image_url:
            raise HTTPException(status_code=500, detail="Failed to upload images to Cloudinary.")

        # Kjør Replicate-modell
        output = replicate.run(
            MODEL,
            input={
                "prompt": prompt,
                "input_image_1": user_image_url,
                "input_image_2": product_image_url
            }
        )

        # Returner bilde-url(er)
        urls = [str(item) for item in output] if isinstance(output, list) else [str(output)]
        return {"image_urls": urls}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
