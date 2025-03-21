import torch
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler
)
from safetensors.torch import load_file
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from PIL import Image, ImageFilter
from io import BytesIO
import cv2


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


class AdvancedInpaintingPipeline:
    def __init__(self, device="cuda"):
        self.device = device

        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)

        # Load Inpainting Pipeline
        self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            "weights/diffusion_checkpoints/checkpoint.safetensors",
            controlnet=self.controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        # Enable memory optimizations
        self.inpaint_pipe.enable_model_cpu_offload()
        self.inpaint_pipe.enable_vae_slicing()

        # Use DDIM scheduler for better quality
        self.inpaint_pipe.scheduler = DDIMScheduler.from_config(
            self.inpaint_pipe.scheduler.config
        )

        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # Initialize image
        self.source_image = None

    def preprocess_image(self, image: Image.Image, target_size: int):
        """Preprocess the input image with custom size."""
        # Resize while maintaining aspect ratio
        ratio = target_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)

        # Create square canvas
        new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_pos = ((target_size - new_size[0]) // 2,
                     (target_size - new_size[1]) // 2)
        new_image.paste(image, paste_pos)

        # Create a control image
        # control_image = make_canny_condition(new_image)
        return new_image  # , control_image

    def preprocess_mask(self, mask: Image.Image, target_size: int):
        """Preprocess the mask with custom size."""
        mask = mask.convert('L')
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        mask = np.array(mask) > 127.5
        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        return mask

    # TODO: Đây là đoạn code để chỉnh kích thước ảnh đầu ra nhưng bị lỗi cần phải sửa
    def postprocess_image(self, image: Image.Image, original_size: tuple, output_size: tuple = None):
        """
        Postprocess the image to match desired output size.

        Args:
            image: Generated image
            original_size: Original input image size (width, height)
            output_size: Desired output size (width, height), if None uses original_size
        """
        # Get the target size
        target_size = output_size if output_size else original_size

        # First crop to remove padding
        if image.size != original_size:
            # Calculate padding
            pad_x = (image.size[0] - original_size[0]) // 2
            pad_y = (image.size[1] - original_size[1]) // 2

            # Crop to original aspect ratio
            image = image.crop((
                pad_x,
                pad_y,
                pad_x + original_size[0],
                pad_y + original_size[1]
            ))

        # Resize to target size if different
        if image.size != target_size:
            image = image.resize(target_size, Image.LANCZOS)

        return image

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        control_image: Image.Image | None = None,
        output_size: tuple | None = None,
        model_size: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        eta: float = 1.0,
        controlnet_conditioning_scale: float = 0.2,
        num_samples: int = 1
    ):
        """
        Enhanced inpainting function with size control.

        Args:
            image: Input image
            mask: Input mask
            prompt: Inpainting prompt
            output_size: Desired output size as (width, height) tuple
            model_size: Size for internal model processing (default 1024 for SDXL)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for stable diffusion
            num_samples: Number of samples to generate
        """
        # Store original size
        original_size = image.size

        # Preprocess inputs to model size
        processed_image = self.preprocess_image(image, model_size)
        processed_mask = self.preprocess_mask(mask, model_size)
        enhanced_prompt = self.enhance_prompt(prompt)

        results = []
        scores = []

        if control_image == None:
            control_image = processed_image
        else:
            control_image = control_image.resize((processed_image.width,
                                                 processed_image.height), Image.LANCZOS)

        for _ in range(num_samples):
            # Generate inpainting
            output = self.inpaint_pipe(
                prompt=enhanced_prompt,
                image=processed_image,
                mask_image=processed_mask,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=torch.manual_seed(np.random.randint(0, 1000000))
            )

            result = output.images[0]

            # Postprocess to desired size
            # TODO: Đây là đoạn code để chỉnh kích thước ảnh đầu ra nhưng bị lỗi cần phải sửa
            # result = self.postprocess_image(result, original_size, output_size)

            # Calculate CLIP score
            clip_score = self.get_clip_score(result, prompt)

            results.append(result)
            scores.append(clip_score)

        # Return best result based on CLIP score
        best_idx = np.argmax(scores)
        return results[best_idx], scores[best_idx]

    def get_clip_score(self, image: Image.Image, prompt: str):
        """Calculate CLIP score between image and prompt."""
        # Move model to GPU temporarily
        self.clip_model.to(self.device)

        inputs = self.clip_processor(
            images=image,
            text=[prompt],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            score = outputs.logits_per_image[0].item()

        # Move model back to CPU
        self.clip_model.cpu()
        torch.cuda.empty_cache()  # Clear GPU memory

        return score

    def post_process(self, result: Image.Image, original: Image.Image):
        """Apply post-processing to improve the result."""
        # Convert to tensor
        result_tensor = torch.from_numpy(np.array(result)).float() / 255.0
        original_tensor = torch.from_numpy(np.array(original)).float() / 255.0

        # Apply color matching
        result_tensor = self.match_color_distribution(
            result_tensor,
            original_tensor
        )

        return Image.fromarray(
            (result_tensor.numpy() * 255).astype(np.uint8)
        )

    def enhance_prompt(self, prompt: str):
        """Enhance the prompt for better results."""
        enhancements = [
            "high quality",
            "detailed",
            "realistic",
            "seamless integration"
        ]
        return f"{prompt}, {', '.join(enhancements)}"

    def match_color_distribution(self, source, target):
        """Match color distribution of source to target."""
        # Calculate mean and std for each channel
        for c in range(3):
            mean_t = torch.mean(target[..., c])
            std_t = torch.std(target[..., c])
            mean_s = torch.mean(source[..., c])
            std_s = torch.std(source[..., c])

            # Apply color matching
            source[..., c] = ((source[..., c] - mean_s)
                              * (std_t / std_s)) + mean_t

        return torch.clamp(source, 0, 1)

    def set_image(self, image_bytes: bytes):
        self.source_image = Image.open(BytesIO(image_bytes)).convert("RGB")


inpainting_pipeline = AdvancedInpaintingPipeline()
