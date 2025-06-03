from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Union
import replicate
import os

# 🚀 1. Start FastAPI først
app = FastAPI()

# 🔒 2. Last inn miljøvariabler fra .env
load_dotenv()

# 🧭 3. Sjekk at API-token og frontend-opprinnelse finnes
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt. Sjekk .env eller Render-miljøet.")
if not FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN er ikke satt. Sjekk .env eller Render-miljøet.")

# 🌐 4. Legg til CORS etter at app er definert
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 5. Modellreferanse
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# ✏️ 6. Definer datastrukturer
class Prompt(BaseModel):
    prompt: str

class ImageResponse(BaseModel):
    image_urls: List[str]

# 🎨 7. Endepunkt for bildegenerering
@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        raw_output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={"prompt": data.prompt},
        )

        # Hjelpefunksjon for å trekke ut URL-er
        def extract_url(item):
            return getattr(item, "url", str(item))

        urls: List[str] = (
            [extract_url(item) for item in raw_output] if isinstance(raw_output, list)
            else [extract_url(raw_output)]
        )

        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
