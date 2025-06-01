from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware

# Start FastAPI-appen tidlig
app = FastAPI()

# CORS settings – frontend-URL fra nettleser
origins = [
    "https://verbose-fiesta-v65pqq949jwp3pp5j-5173.app.github.dev/",  # ← bytt ut ved behov
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # eller ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Last inn miljøvariabler
load_dotenv()

# Hent API-token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt. Sjekk .env eller Render settings.")

# Modellversjon
MODEL_REF = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"

# Inndata
class Prompt(BaseModel):
    prompt: str

# Utdata
class ImageResponse(BaseModel):
    image_urls: List[str]

# Endpoint
@app.post("/generate", response_model=ImageResponse)
def generate_image(data: Prompt):
    try:
        raw_output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={"prompt": data.prompt},
            api_token=REPLICATE_API_TOKEN,
        )

        def extract_url(item):
            return item.url if hasattr(item, "url") else str(item)

        urls = [extract_url(item) for item in raw_output] if isinstance(raw_output, list) else [extract_url(raw_output)]

        return {"image_urls": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
