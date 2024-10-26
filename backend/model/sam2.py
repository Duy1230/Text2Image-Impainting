import numpy as np
import base64
from io import BytesIO
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2  # Ensure OpenCV is imported for the new function

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")


def base64_to_image(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    return image


class SAM2:
    def __init__(self, model_name: str = "facebook/sam2-hiera-tiny"):
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)

    def set_image(self, image_bytes: bytes):
        """
        Set the image from raw bytes.

        Parameters:
        - image_bytes (bytes): The image data in bytes.
        """
        try:
            self.image = Image.open(BytesIO(image_bytes))
            self.predictor.set_image(self.image)
        except Exception as e:
            raise ValueError(f"Failed to set image: {e}")

    def segment(self, coordinate: np.ndarray, label: np.ndarray):
        image_width, image_height = self.image.size
        coordinate = coordinate * np.array([image_width, image_height])
        masks, scores, logits = self.predictor.predict(
            point_coords=coordinate.astype(np.int16),
            point_labels=label,
            multimask_output=False
        )
        self.masks = masks
        self.scores = scores
        self.logits = logits
        return coordinate

    def segment_from_boxes(self, boxes: np.ndarray):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        self.masks = masks
        self.scores = scores
        self.logits = logits
        return boxes

    def show_mask(self, random_color=False, borders=True):
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = self.masks.shape[-2:]
        mask = self.masks.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(
                self.masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(
                contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(
                mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        return mask_image

    def apply_bluer_mask(self, alpha: float = 0.5) -> np.ndarray:
        """
        Apply a semi-transparent blue overlay to the regions of the image where the mask is 1.

        Parameters:
        - alpha (float): Transparency factor for the blue overlay. Default is 0.5.

        Returns:
        - np.ndarray: The RGB image with the blue overlay applied where mask == 1.
        """
        # Ensure self.image is a NumPy array
        if isinstance(self.image, Image.Image):
            image_np = np.array(self.image)
        elif isinstance(self.image, np.ndarray):
            image_np = self.image
        else:
            raise TypeError("self.image must be a PIL Image or a NumPy array")

        if self.masks.shape[0] != 1:
            raise ValueError("Mask should have shape (1, H, W)")
        if image_np.shape[2] != 3:
            raise ValueError("Image should have shape (H, W, 3)")
        if self.masks.shape[1:] != image_np.shape[:2]:
            raise ValueError("Mask and image spatial dimensions must match")
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")

        # Copy the image to avoid modifying the original
        blended = image_np.copy().astype(np.float32)

        # Create a blue overlay
        blue_overlay = np.zeros_like(blended)
        blue_overlay[:, :, 2] = 255  # Set blue channel to maximum

        # Expand mask to match image channels
        mask_expanded = self.masks[0, :, :, np.newaxis]  # Shape: (H, W, 1)

        # Apply the overlay only where mask == 1
        blended = np.where(
            mask_expanded == 1,
            (1 - alpha) * blended + alpha * blue_overlay,
            blended
        )

        # Ensure the pixel values are in the valid range
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended


sam2_model = SAM2()
