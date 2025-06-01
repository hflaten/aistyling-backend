from fastapi import FastAPI
from pydantic import BaseModel
import os
import replicate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Sett opp Replicate-klienten med eksplisitt token
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: Prompt):
    try:
        output = replicate_client.run(
            "stability-ai/stable-diffusion:latest",
            input={"prompt": data.prompt}
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}
