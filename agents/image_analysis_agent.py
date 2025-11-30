# agents/image_analysis_agent.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["Normal", "Pneumonia", "Other"]  # نمونه کلاس‌ها

class ImageAnalysisAgent:
    """
    This agent is intended for demonstration/Capstone.
    For real competition, train properly and provide trained weights in models/chest_classifier.pt.
    If models/chest_classifier.pt exists, it will try to load it (state_dict for fc layer).
    Otherwise, uses ImageNet-pretrained ResNet18 backbone and a fresh fc layer.
    """

    def __init__(self, model_path="models/chest_classifier.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # build model
        backbone = models.resnet18(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        # try to load weights if provided
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                print("Loaded model weights from", model_path)
            except Exception as e:
                print("Warning: failed to load model weights:", e)

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self, input_data):
        # input_data may be a path string or dict with 'user_request'
        img_path = input_data if isinstance(input_data, str) else (
            input_data.get("user_request") if isinstance(input_data, dict) else None
        )
        if img_path is None:
            return {"pathology": "", "confidence": "0%"}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return {"pathology": "ErrorOpen", "confidence": "0%", "error": str(e)}

        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = CLASS_NAMES[idx]
            conf = float(probs[idx])

        return {"pathology": label, "confidence": f"{conf:.2f}"}
