from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import File, Form, UploadFile
from io import BytesIO
from PIL import Image
from model.sam2 import sam2_model
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/segment")
async def segment_image(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    # segment image using SAM2 model
    sam2_model.segment(np.array([[x, y]]), np.array([1]))

    # apply blue mask to image
    applied_mask = sam2_model.apply_bluer_mask(image)

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(applied_mask.astype('uint8'), 'RGB')

    # Create a byte stream to hold the image data
    img_byte_arr = BytesIO()

    # Save the image as PNG to the byte stream
    pil_image.save(img_byte_arr, format='PNG')

    # Get the byte string from the stream
    img_byte_arr = img_byte_arr.getvalue()

    # Return the image as a binary response
    return Response(content=img_byte_arr, media_type="image/png")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@router.post("/add-image")
async def add_image(image: UploadFile = File(...)):
    """
    Endpoint to add an image to the SAM2 model.

    Parameters:
    - image (UploadFile): The image file uploaded by the user.

    Returns:
    - JSONResponse: A success message or an error message.
    """
    try:
        # Validate the uploaded file
        file_extension = image.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail="Invalid image format.")

        # Read the image bytes
        image_bytes = await image.read()
        await image.close()  # Close the file after reading

        # Set the image in the SAM2 model
        sam2_model.set_image(image_bytes)

        logger.info("Image added successfully.")
        return JSONResponse(content={"message": "Image added successfully"}, status_code=200)
    except HTTPException as http_err:
        logger.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Error adding image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
