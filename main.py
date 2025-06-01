from fastapi import FastAPI
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv

load_dotenv()  # Laster miljøvariabler fra .env

app = FastAPI()

replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: Prompt):
    output = replicate.run(
        "stability-ai/stable-diffusion:latest",
        input={"prompt": data.prompt}
    )
    return {"image_url": output}
