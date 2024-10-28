from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image
from io import BytesIO
import numpy as np

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


class GroundingDINO:
    def __init__(self):
        self.model = load_model("weights/groundingdino/GroundingDINO_SwinT_OGC.py",
                                "weights/groundingdino/groundingdino_swint_ogc.pth")
        self.boxes = None
        self.logits = None
        self.phrases = None

    def set_image(self, image_bytes: bytes):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        self.image_transformed = image_transformed

    def predict(self, prompt: str, single_target_mode: bool):
        boxes, logits, phrases = predict(
            model=self.model,
            image=self.image_transformed,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        if single_target_mode:
            max_idx = logits.argmax()
            self.boxes = boxes[max_idx].unsqueeze(0)  # Keep tensor format with single box
            self.logits = logits[max_idx].unsqueeze(0)  # Keep tensor format with single logit
            self.phrases = [phrases[max_idx]]  # Phrases is already a list
        else:
            self.boxes = boxes
            self.logits = logits
            self.phrases = phrases

    def get_boxes(self):
        return self.boxes.tolist()

    def get_logits(self):
        return self.logits.tolist()

    def get_phrases(self):
        return self.phrases


groundingdino_model = GroundingDINO()
