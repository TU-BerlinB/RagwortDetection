import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv3:
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