from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List, Union
import replicate
import os
import tempfile
import shutil

# 🚀 Start FastAPI
app = FastAPI()

# 🔐 Last inn miljøvariabler
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")

if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN er ikke satt.")
if not FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN er ikke satt.")

# 🌐 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Modellreferanse
MODEL_REF = "flux-kontext-apps/multi-image-kontext-max"

# 🧰 Hjelpefunksjon: midlertidig lagre bilde
def save_upload_file_temp(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

# 🎯 Endepunkt for bildegenerering
@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    image: UploadFile = Form(...),
    context_image: UploadFile = Form(...),
):
    try:
        image_path = save_upload_file_temp(image)
        context_path = save_upload_file_temp(context_image)

        output: Union[List, object] = replicate.run(
            MODEL_REF,
            input={
                "prompt": prompt,
                "image": open(image_path, "rb"),
                "context_image": open(context_path, "rb"),
            },
        )

        def extract_url(item):
            return getattr(item, "url", str(item))

        urls: List[str] = (
            [extract_url(item) for item in output] if isinstance(output, list)
            else [extract_url(output)]
        )

        return JSONResponse(content={"image_urls": urls})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
