from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.cloudinary_utils import upload_image_to_cloudinary
from pipeline.flux_pipeline import run_flux_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # I prod: angi frontend-URL
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
        user_image_url = await upload_image_to_cloudinary(user_image)
        product_image_url = await upload_image_to_cloudinary(product_image)
        image_url = run_flux_pipeline(user_image_url, product_image_url, prompt)
        return {"image_urls": [image_url]}
    except Exception as e:
        return {"detail": str(e)}
