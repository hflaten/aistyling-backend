import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils.cloudinary_utils import upload_image_to_cloudinary, upload_pil_image_to_cloudinary
from utils.combine_images import combine_images_from_urls
import replicate

app = FastAPI()
load_dotenv()


# CORS-oppsett
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replicate API
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

@app.post("/generate")
async def generate_image(
    user_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # 1. Last opp bruker- og produktbilde
        user_url = upload_image_to_cloudinary(user_image)
        product_url = upload_image_to_cloudinary(product_image)

        # 2. Kombiner bilder og last opp
        combined_img = combine_images_from_urls(user_url, product_url)
        combined_url = upload_pil_image_to_cloudinary(combined_img)

        # 3. Kjør modell
        output = replicate_client.run(
            "black-forest-labs/flux-kontext-max",
            input={
                "input_image": combined_url,
                "prompt": prompt,
                "output_format": "jpg",
                "aspect_ratio": "match_input_image",
                "safety_tolerance": 2
            }
        )

        return {"output_url": output}

    except Exception as e:
        return {"detail": str(e)}
