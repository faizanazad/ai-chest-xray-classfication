from pathlib import Path
import io
import logging
import pickle

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chest X-Ray Classification Demo")

templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Existing trained model: no retraining
model = tf.keras.models.load_model(
    "models/CNN_Covid19_Xray_version.h5",
    compile=False
)

with open("models/Lable_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

IMAGE_SIZE = 150
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Check that the model and encoder agree
if model.input_shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
    raise ValueError(f"Unexpected model input shape: {model.input_shape}")

if model.output_shape[-1] != len(label_encoder.classes_):
    raise ValueError("Model output count does not match encoder classes.")

logger.info("Model input: %s", model.input_shape)
logger.info("Model output: %s", model.output_shape)
logger.info("Classes: %s", label_encoder.classes_)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="camera.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        # Read file with a size limit
        image_data = await file.read(MAX_FILE_SIZE + 1)

        if not image_data:
            raise ValueError("Please select an image first.")

        if len(image_data) > MAX_FILE_SIZE:
            raise ValueError("Image must be smaller than 10 MB.")

        # Decode and validate image
        with Image.open(io.BytesIO(image_data)) as uploaded_image:
            if uploaded_image.format not in ALLOWED_FORMATS:
                raise ValueError("Please upload a JPEG, PNG, or WEBP image.")

            uploaded_image.verify()

        # Reopen after verify, then preprocess
        with Image.open(io.BytesIO(image_data)) as uploaded_image:
            image = uploaded_image.convert("RGB")
            #preliminary filter for highly colorful images
            hsv_array = np.array(image.convert("HSV"))
            mean_saturation = float(hsv_array[:,:,1].mean())
            if mean_saturation > 45:
                raise ValueError("Invalid image. Please upload a valid chest X-Ray image.")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

            image_array = np.asarray(image, dtype=np.float32) / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = model.predict(image_array, verbose=0)[0]

        if (
            len(predictions) != len(label_encoder.classes_)
            or not np.all(np.isfinite(predictions))
        ):
            raise ValueError("The model returned an invalid prediction.")

        predicted_index = int(np.argmax(predictions))
        result = str(
            label_encoder.inverse_transform([predicted_index])[0]
        )
        confidence = float(predictions[predicted_index] * 100)

        # Scores are model outputs, not disease probabilities
        class_scores = [
            {
                "label": str(label),
                "score": round(float(score) * 100, 2)
            }
            for label, score in zip(
                label_encoder.classes_,
                predictions
            )
        ]

        logger.info(
            "Prediction completed. Class=%s, score=%.2f%%",
            result,
            confidence
        )

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "result": result,
                "confidence": round(confidence, 2),
                "class_scores": class_scores,
                "error": None
            }
        )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "result": None,
                "confidence": None,
                "class_scores": [],
                "error": str(exc) or "This image could not be processed."
            },
            status_code=400
        )

    except Exception:
        logger.exception("Prediction failed")
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "result": None,
                "confidence": None,
                "class_scores": [],
                "error": "Prediction failed. Please try another image."
            },
            status_code=500
        )

    finally:
        await file.close()

