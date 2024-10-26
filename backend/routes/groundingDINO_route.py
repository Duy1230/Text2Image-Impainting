from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import File, Form, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from model.groundingDINO import groundingdino_model
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    return {"message": "This is the GroundingDINO route"}


class GroundingDINORequest(BaseModel):
    prompt: str


@router.post("/predict")
async def predict(request: GroundingDINORequest):
    groundingdino_model.predict(request.prompt)

    print("boxes: ", groundingdino_model.get_boxes())
    print("logits: ", groundingdino_model.get_logits())
    print("phrases: ", groundingdino_model.get_phrases())

    return JSONResponse(content={"boxes": groundingdino_model.get_boxes(), "logits": groundingdino_model.get_logits(), "phrases": groundingdino_model.get_phrases()})


@router.post("/set_image")
async def set_image(image: UploadFile = File(...)):
    """
    Endpoint to add an image to the SAM2 model.

    Parameters:
    - image (UploadFile): The image file uploaded by the user.

    Returns:
    - JSONResponse: A success message or an error message.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    try:
        # validate the image
        file_extension = image.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail="Invalid image format.")

        # read the image
        image_bytes = await image.read()
        await image.close()

        groundingdino_model.set_image(image_bytes)
        logger.info("Image set successfully.")
        return JSONResponse(content={"message": "Image set successfully"}, status_code=200)
    except Exception as e:
        logger.error(f"Error setting image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
