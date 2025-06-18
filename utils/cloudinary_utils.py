import cloudinary
import os
import cloudinary.uploader
from fastapi import UploadFile
from PIL import Image
from io import BytesIO

# Konfigurer Cloudinary med miljøvariabler
def configure_cloudinary():
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )

configure_cloudinary()

def upload_image_to_cloudinary(upload_file: UploadFile) -> str:
    """
    Laster opp et bilde fra FastAPI UploadFile til Cloudinary og returnerer URL.
    """
    contents = upload_file.file.read()
    upload_result = cloudinary.uploader.upload(BytesIO(contents), resource_type="image")
    return upload_result["secure_url"]

def upload_pil_image_to_cloudinary(image: Image.Image) -> str:
    """
    Laster opp et PIL-bilde til Cloudinary og returnerer URL.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    upload_result = cloudinary.uploader.upload(buffer, resource_type="image")
    return upload_result["secure_url"]
