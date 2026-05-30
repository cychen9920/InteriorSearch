from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from PIL.Image import Image as PilImage


class ClipEncoder:
    """Small PyTorch/OpenCLIP wrapper for image and text embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        images = [self._load_image(path) for path in image_paths]
        return self._encode_image_tensors(images)

    @torch.inference_mode()
    def encode_pil_image(self, image: PilImage) -> np.ndarray:
        image_tensor = self.preprocess(image.convert("RGB"))
        features = self._encode_image_tensors([image_tensor])
        return features[0]

    @torch.inference_mode()
    def _encode_image_tensors(self, images: list[torch.Tensor]) -> np.ndarray:
        image_tensor = torch.stack(images).to(self.device)
        features = self.model.encode_image(image_tensor)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)[0]

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))
