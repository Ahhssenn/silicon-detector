import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


class SiliconClassifier:
    """
    Small helper class to load the trained silicon model
    and run predictions on single images.
    """

    def __init__(self, model_path: str = "silicon_model.pth", device: str | None = None, threshold: float = 0.5):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Select device
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        classes = checkpoint.get("classes")
        if classes is None:
            # Default to binary classes if not stored
            classes = ["non_silicon", "silicon"]
        self.classes = classes

        # Build model and load weights
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.threshold = float(threshold)

        # Standard transforms (same normalization as training)
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _prepare_tensor(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        x = self.transforms(img).unsqueeze(0)  # shape: [1, C, H, W]
        return x.to(self.device)

    @torch.no_grad()
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Returns (label, confidence) for a single image.
        Confidence is the softmax probability of the predicted class.
        """
        x = self._prepare_tensor(image_path)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        idx = int(probs.argmax())
        label = self.classes[idx]
        confidence = float(probs[idx])
        return label, confidence

    @torch.no_grad()
    def is_silicon(self, image_path: str, silicon_label: str = "silicon") -> bool:
        """
        Convenience method: returns True if the image is classified as 'silicon'
        with probability above the configured threshold.
        """
        label, confidence = self.predict(image_path)
        return label == silicon_label and confidence >= self.threshold
