import replicate

model = replicate.models.get("fluxkontext-apps/multi-image-kontext-max")
version = model.versions.get("6b5cabb2c020b65cfa8c5564903794f8373395f474d20e5e80f8d49b0aa30418")

def run_flux_pipeline(user_image_url: str, product_image_url: str, prompt: str) -> str:
    input_data = {
        "input_image_1": user_image_url,
        "input_image_2": product_image_url,
        "prompt": prompt
    }

    output = version.predict(**input_data)

    if isinstance(output, list) and len(output) > 0:
        return output[0]
    else:
        raise Exception("Model did not return a valid image URL.")
