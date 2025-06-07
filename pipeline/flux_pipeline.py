import replicate

model = replicate.models.get("black-forest-labs/flux-kontext-max")


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
