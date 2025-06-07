import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

async def upload_image_to_cloudinary(file):
    contents = await file.read()
    upload_result = cloudinary.uploader.upload(contents, resource_type="image")
    return upload_result["secure_url"]
