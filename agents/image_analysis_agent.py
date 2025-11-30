# agents/image_analysis_agent.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image, ImageOps

CLASS_NAMES = ["Normal", "Pneumonia", "Other"]

class ImageAnalysisAgent:
    def __init__(self, model_path="models/chest_classifier.pt", device=None, threshold=0.6):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # build model
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        # load weights if available
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, img_path):
        try:
            img = Image.open(img_path).convert("L")  # grayscale
            img = ImageOps.equalize(img)  # improve contrast
            img = img.convert("RGB")
            return self.tf(img).unsqueeze(0).to(self.device)
        except Exception as e:
            return None, str(e)

    def run(self, input_data):
        img_path = input_data if isinstance(input_data, str) else (
            input_data.get("user_request") if isinstance(input_data, dict) else None
        )
        if img_path is None:
            return {"pathology": "", "confidence": "0%"}

        x, error = self.preprocess(img_path), None
        if x is None:
            return {"pathology": "ErrorOpen", "confidence": "0%", "error": error}

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = CLASS_NAMES[idx]

        if conf < self.threshold:
            label = "Other"

        return {"pathology": label, "confidence": f"{conf:.2f}"}
