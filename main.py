from fastapi import FastAPI
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Sett API-token
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate_image(data: Prompt):
    try:
        output = replicate.run(
            "stability-ai/stable-diffusion:latest",
            input={"prompt": data.prompt}
        )
        return {"image_url": output}
    except Exception as e:
        return {"error": str(e)}
