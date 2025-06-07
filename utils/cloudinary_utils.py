import cloudinary
import cloudinary.uploader
from fastapi import UploadFile
import os

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

async def upload_image_to_cloudinary(file: UploadFile) -> str:
    contents = await file.read()
    result = cloudinary.uploader.upload(contents, resource_type="image")
    return result["secure_url"]
