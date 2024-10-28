from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import File, Form, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageFilter
from model.diffusion_pipline import inpainting_pipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    return {"message": "This is the Diffusion route"}


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

        inpainting_pipeline.set_image(image_bytes)
        logger.info("Diffusion route: Image set successfully.")
        return JSONResponse(content={"message": "Diffusion route: Image set successfully"}, status_code=200)
    except Exception as e:
        logger.error(f"Diffusion route: Error setting image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class InpaintingRequest(BaseModel):
    prompt: str
    mask: list  # Change this to accept a list instead of np.ndarray

    class Config:
        arbitrary_types_allowed = True


def convert_binary_mask_to_PIL(mask: np.ndarray):
    """Convert binary mask to PIL Image.

    Args:
        mask: Binary mask of shape (1, H, W) containing only 0s and 1s

    Returns:
        PIL Image with shape (H, W, 3) where 1s are converted to 255
    """
    # Remove single channel dimension and expand to 3 channels
    mask = mask.squeeze(0)  # (H, W)
    mask = np.stack([mask] * 3, axis=-1)  # (H, W, 3)

    # Convert 1s to 255
    mask = (mask * 255).astype(np.uint8)

    return Image.fromarray(mask)


@router.post("/inpainting")
async def inpainting(request: InpaintingRequest):
    prompt = request.prompt
    source = inpainting_pipeline.source_image
    # Convert the list back to numpy array
    mask_array = np.array(request.mask)
    mask = convert_binary_mask_to_PIL(mask_array)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=10))

    result, clip_score = inpainting_pipeline.inpaint(
        image=source,
        mask=mask,
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=10,
        num_samples=2  # Generate 3 samples and pick the best
    )

    final_result = inpainting_pipeline.post_process(result, source)

    # Create a byte stream to hold the image data
    img_byte_arr = BytesIO()

    # Save the image as PNG to the byte stream
    final_result.save(img_byte_arr, format='PNG')

    # Get the byte string from the stream
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")
