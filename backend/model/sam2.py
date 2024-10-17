import numpy as np
import base64
from io import BytesIO
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")


def base64_to_image(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    return image


class SAM2:
    def __init__(self, model_name: str = "facebook/sam2-hiera-tiny"):
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)

    def set_image(self, base64_string):
        self.image = base64_to_image(base64_string)
        self.predictor.set_image(self.image)

    def segment(self, coordinate: np.ndarray, label: np.ndarray):
        masks, scores, logits = self.predictor.predict(
            point_coords=coordinate,
            point_labels=label,
        )
        self.masks = masks
        self.scores = scores
        self.logits = logits

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
            import cv2
            contours, _ = cv2.findContours(
                self.masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
        contours = [cv2.approxPolyDP(
            contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        return mask_image
