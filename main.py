from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Union

# Last inn miljøvariabler fra .env
load_dotenv()

# Start FastAPI-appen
app = FastAPI()

# Hent frontend origin fra miljøvariabel
frontend_origin = os.getenv("FRONTEND_ORIGIN")
if not frontend_origin:
    raise RuntimeError("FRONTEND_ORIGIN is not set. Legg til i .env")

# CORS-konfigurasjon
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],  # Bare din frontend får tilgang
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hent og valider at API-tokenet finnes
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt. Sjekk .env eller miljøvariabler.")

# Replicate-modellreferanse
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# Inndata-schema
class Prompt(BaseModel):
    prompt: str

# Utdata-schema
class ImageResponse(BaseModel):
    image_urls: List[str]

@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        # Kjør Replicate-modellen
        raw_output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={"prompt": data.prompt},
            api_token=REPLICATE_API_TOKEN,
        )

        # Ekstraher URL-er
        def extract_url(item):
            return getattr(item, "url", str(item))

        if isinstance(raw_output, list):
            urls = [extract_url(item) for item in raw_output]
        else:
            urls = [extract_url(raw_output)]

        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
