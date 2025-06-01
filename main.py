from fastapi import FastAPI
from pydantic import BaseModel
import os
import replicate
from dotenv import load_dotenv

# Last inn API-nøkkelen fra .env-filen
load_dotenv()

# Initialiser FastAPI og Replicate-klienten
app = FastAPI()
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Definer dataformatet vi tar imot fra brukeren
class Prompt(BaseModel):
    prompt: str

# Endepunkt for å generere bilde med Stable Diffusion
@app.post("/generate")
def generate_image(data: Prompt):
    try:
        output = replicate_client.run(
            "stability-ai/stable-diffusion@ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={
                "prompt": data.prompt,
                "scheduler": "K_EULER"
            }
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}
