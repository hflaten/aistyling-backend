import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import replicate
from cloudinary_utils import upload_image_to_cloudinary

app = FastAPI()

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("Missing Replicate API token in environment variables.")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)



origins = [
    "http://localhost",
    "http://localhost:5173",
    os.getenv("FRONTEND_ORIGIN")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...)
):
    try:
        # Upload both images to Cloudinary and get URLs
        user_image_url = await upload_image_to_cloudinary(user_image)
        product_image_url = await upload_image_to_cloudinary(product_image)

        # Send to Replicate model
        output = replicate_client.run(
            "black-forest-labs/flux-kontext-max",
            input={
                "input_image_1": user_image_url,
                "input_image_2": product_image_url,
                "prompt": prompt
            }
        )

        if isinstance(output, list):
            return {"image_urls": output}
        elif isinstance(output, str):
            return {"image_urls": [output]}
        else:
            return JSONResponse(status_code=500, content={"detail": "Unexpected output format from model"})

    except Exception as e:
        print("Image generation failed:", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
