from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
import os
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bytt ut med frontend-URL i produksjon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware


# Last inn miljøvariabler fra .env (lokalt) eller Render-miljø
load_dotenv()

# Start FastAPI-appen
app = FastAPI()
origins = [
    "https://verbose-fiesta-v65pqq949jwp3pp5j-5173.app.github.dev/",  # ← bytt ut med nøyaktig domenet fra nettleseren din
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # eller ["*"] midlertidig for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Hent og valider at API-tokenet finnes
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt. Sjekk .env eller miljøvariabler.")

# Eksakt modellreferanse fra Replicate (versjon «ac732…» som du viste)
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# Inndata-schema: kun én nøkkel "prompt"
class Prompt(BaseModel):
    prompt: str

# Utdata-schema: én liste med URL-strenger
class ImageResponse(BaseModel):
    image_urls: List[str]

@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        # Kaller modellen med token (hentes direkte fra miljøet)
        raw_output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={"prompt": data.prompt},
            api_token=REPLICATE_API_TOKEN,
        )

        # raw_output kan være enten:
        #  - en liste av FileOutput‐objekter
        #  - ett enkelt FileOutput‐objekt
        # Vi normaliserer til en liste av URL-strenger:
        urls: List[str] = []

        # Hjelpefunksjon for å trekke ut URL‐strengen
        def extract_url(item):
            # FileOutput‐objekt har attributtet `.url`:
            if hasattr(item, "url"):
                return item.url
            # Hvis det er en streng allerede (ikke vanlig i Replicate), la den stå
            return str(item)

        if isinstance(raw_output, list):
            for item in raw_output:
                urls.append(extract_url(item))
        else:
            urls.append(extract_url(raw_output))

        return {"image_urls": urls}

    except Exception as e:
        # Hvis noe feiler (f.eks. 404, 401, etc.), kast HTTP 500
        raise HTTPException(status_code=500, detail=str(e))
