from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import File, Form, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageFilter
from model.diffusion_pipline import inpainting_pipeline, make_canny_condition
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


def convert_binary_mask_to_PIL(mask: np.ndarray):
    """Convert mask to binary mask and then to PIL Image.

    Args:
        mask: Mask array of shape (1, H, W). If not binary, will be thresholded at 0.5

    Returns:
        PIL Image with shape (H, W, 3) where 1s are converted to 255
    """
    # Ensure mask is binary
    if not np.array_equal(mask, mask.astype(bool)):
        # If mask contains values other than 0 and 1, threshold at 0.5
        mask = (mask > 0.5).astype(np.float32)

    # Remove single channel dimension and expand to 3 channels
    mask = mask.squeeze(0)  # (H, W)
    mask = np.stack([mask] * 3, axis=-1)  # (H, W, 3)
    # Convert 1s to 255
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask).convert("RGB")


class InpaintingRequest(BaseModel):
    prompt: str
    mask: list  # Change this to accept a list instead of np.ndarray
    postprocess_mode: bool
    is_applying_blur: bool
    using_canny_control_image: bool
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    num_samples: int
    mask_rescale: float

    class Config:
        arbitrary_types_allowed = True


def resize_and_crop_center(image: Image.Image, k: float) -> Image.Image:
    """
    Resize image by factor k and handle the result to match original size.
    If k > 1: crops the center region
    If k < 1: pads the borders with black

    Args:
        image: PIL Image object
        k: scaling factor (must be > 0)

    Returns:
        PIL Image with original size
    """
    if k <= 0:
        raise ValueError("Scale factor k must be greater than 0")
    # Get original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    new_width = int(original_width * k)
    new_height = int(original_height * k)

    # Resize image
    resized_image = image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    if k >= 1:
        # Calculate crop coordinates
        left = (new_width - original_width) // 2
        top = (new_height - original_height) // 2
        right = left + original_width
        bottom = top + original_height

        # Crop center region
        result_image = resized_image.crop((left, top, right, bottom))
    else:
        # Create black background
        result_array = np.zeros(
            (original_height, original_width, 3), dtype=np.uint8)
        result_image = Image.fromarray(result_array)

        # Calculate paste coordinates
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2

        # Paste resized image onto black background
        result_image.paste(resized_image, (left, top))

    return result_image


@router.post("/inpainting")
async def inpainting(request: InpaintingRequest):
    prompt = request.prompt
    source = inpainting_pipeline.source_image
    # Convert the list back to numpy array
    mask_array = np.array(request.mask)
    mask = convert_binary_mask_to_PIL(mask_array)
    mask = resize_and_crop_center(mask, request.mask_rescale)
    if request.is_applying_blur:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
    if request.using_canny_control_image:
        control_image = make_canny_condition(source)
    else:
        control_image = None

    result, clip_score = inpainting_pipeline.inpaint(
        image=source,
        mask=mask,
        prompt=prompt,
        control_image=control_image,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        controlnet_conditioning_scale=request.controlnet_conditioning_scale,
        num_samples=request.num_samples  # Generate 3 samples and pick the best
    )

    if request.postprocess_mode:
        final_result = inpainting_pipeline.post_process(result, source)
    else:
        final_result = result

    # Create a byte stream to hold the image data
    img_byte_arr = BytesIO()

    # Save the image as PNG to the byte stream
    final_result.save(img_byte_arr, format='PNG')

    # Get the byte string from the stream
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")
