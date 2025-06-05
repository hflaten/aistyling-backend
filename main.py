from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv
import replicate
import os

# 1. Last miljøvariabler og init app
load_dotenv()
app = FastAPI()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
if not REPLICATE_API_TOKEN or not FRONTEND_ORIGIN:
    raise RuntimeError("Missing required environment variables.")

# 2. CORS-policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Modellreferanse
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# 4. Endepunkt for generering
@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    user_image: Union[UploadFile, None] = File(default=None),
    product_image: Union[UploadFile, None] = File(default=None),
):
    try:
        # ⚙️ Bygg input til modellen
        input_data = {"prompt": prompt}

        if user_image:
            input_data["image"] = await user_image.read()  # image-to-image med brukerbilde
        elif product_image:
            input_data["image"] = await product_image.read()  # image-to-image med produktbilde
        # Hvis ingen bilde: kjør ren tekstprompt

        # 🚀 Kjør modell
        raw_output: Union[List, object] = replicate.run(MODEL_REF, input=input_data)

        def extract_url(item):
            return getattr(item, "url", str(item))

        urls: List[str] = (
            [extract_url(item) for item in raw_output] if isinstance(raw_output, list)
            else [extract_url(raw_output)]
        )

        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
