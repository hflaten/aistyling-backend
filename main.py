from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv
import replicate
import os
import tempfile

# 2. Sett opp FastAPI og CORS
app = FastAPI()

# 1. Last inn .env
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt. Sjekk .env")
if not FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN er ikke satt. Sjekk .env")



app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Modellreferanse for Flux
MODEL_REF = "flux-kontext-apps/multi-image-kontext-max"

# 4. Endepunkt for bildegenerering
@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    try:
        # Midlertidig lagre bildene
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as user_temp:
            user_temp.write(await user_image.read())
            user_path = user_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as product_temp:
            product_temp.write(await product_image.read())
            product_path = product_temp.name

        # Kjør modellen
        output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={
                "image_1": open(user_path, "rb"),
                "image_2": open(product_path, "rb"),
                "prompt": prompt,
            },
        )

        # Hent ut URL-er (Replicate returnerer som regel bare én)
        def extract_url(item):
            return getattr(item, "url", str(item))

        urls = (
            [extract_url(item) for item in output] if isinstance(output, list)
            else [extract_url(output)]
        )

        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
