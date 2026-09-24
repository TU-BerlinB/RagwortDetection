"""
File: src/models/dinov3.py
Usage:
    from src.models.dinov3 import DINOv3
    model = DINOv3()
    features = model.extract_features(image)
Description:
    Wrapper class for the DINOv3 Vision Transformer backbone from HuggingFace,
    providing an interface for extracting feature embeddings from input images.
"""

import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv3:
    """Wrapper for DINOv3 Vision Transformer model."""

    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, image):
        """Extract last hidden state feature tokens from an input image or batch of images."""
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state