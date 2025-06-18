import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import replicate
from utils.cloudinary_utils import upload_image_to_cloudinary
from utils.combine_images import combine_images

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
        # Upload original images to Cloudinary
        user_image_url = await upload_image_to_cloudinary(user_image)
        product_image_url = await upload_image_to_cloudinary(product_image)

        # Step 1: Remove background from user image
        remove_bg_output = replicate_client.run(
            "cjwbw/remove-bg",
            input={"image": user_image_url}
        )
        user_image_no_bg = remove_bg_output if isinstance(remove_bg_output, str) else remove_bg_output[0]

        # Step 2: Enhance face with CodeFormer
        face_restore_output = replicate_client.run(
            "sczhou/codeformer",
            input={
                "image": user_image_no_bg,
                "face_upsample": True,
                "codeformer_fidelity": 0.8
            }
        )
        enhanced_user_image = face_restore_output if isinstance(face_restore_output, str) else face_restore_output[0]

        # Step 3: Combine enhanced user image with product image
        composed_image_url = combine_images(enhanced_user_image, product_image_url)

        # Step 4: Final generation with Flux
        final_output = replicate_client.run(
            "black-forest-labs/flux-kontext-max",
            input={
                "input_image": composed_image_url,
                "prompt": prompt
            }
        )

        if isinstance(final_output, list):
            return {"image_urls": final_output}
        elif isinstance(final_output, str):
            return {"image_urls": [final_output]}
        else:
            return JSONResponse(status_code=500, content={"detail": "Unexpected output format from model"})

    except Exception as e:
        print("Image generation failed:", str(e))
        return JSONResponse(status_code=500, content={"detail": str(e)})
