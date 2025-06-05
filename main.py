from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv
import replicate
import os
import logging

# 🚀 1. Start appen
app = FastAPI()

# 🔐 2. Last miljøvariabler
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt.")
if not FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN er ikke satt.")

# 🔄 3. Sett opp CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 4. Modellreferanse (Stable Diffusion 1.5)
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# 🎯 5. Output-modell
class ImageResponse(BaseModel):
    image_urls: List[str]

# 🖼️ 6. Genererings-endepunkt
@app.post("/generate", response_model=ImageResponse)
async def generate_image(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):
    try:
        input_data = {"prompt": prompt}

        if image:
            image_data = await image.read()
            input_data["image"] = image_data

        output: Union[List, object] = replicate.run(
            MODEL_REF,
            input=input_data
        )

        def extract_url(item):
            return getattr(item, "url", str(item))

        urls = (
            [extract_url(item) for item in output]
            if isinstance(output, list)
            else [extract_url(output)]
        )

        return {"image_urls": urls}

    except Exception as e:
        logging.exception("Feil under generering")
        raise HTTPException(status_code=500, detail=str(e))
