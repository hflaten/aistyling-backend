# utils/combine_images.py

from PIL import Image
import requests
from io import BytesIO
from utils.cloudinary_utils import upload_pil_image_to_cloudinary


def combine_images(user_image_url: str, product_image_url: str) -> str:
    # Last ned begge bildene
    response_user = requests.get(user_image_url)
    response_product = requests.get(product_image_url)

    user_image = Image.open(BytesIO(response_user.content)).convert("RGBA")
    product_image = Image.open(BytesIO(response_product.content)).convert("RGBA")

    # Endre størrelse på produktbildet til samme bredde som brukerbildet
    product_image = product_image.resize((user_image.width, int(product_image.height * user_image.width / product_image.width)))

    # Lag en ny canvas for å sette dem vertikalt
    combined_height = user_image.height + product_image.height
    combined_image = Image.new("RGBA", (user_image.width, combined_height))
    combined_image.paste(user_image, (0, 0))
    combined_image.paste(product_image, (0, user_image.height))

    # Last opp kombinert bilde til Cloudinary
    return upload_pil_image_to_cloudinary(combined_image)
