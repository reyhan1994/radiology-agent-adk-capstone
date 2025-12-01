# agents/image_analysis_agent.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["Normal", "Pneumonia", "Other"]

class ImageAnalysisAgent:
    def __init__(self, model_path="models/chest_classifier.pt", threshold=0.45, device=None, tta=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.tta = bool(tta)

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print("✅ Loaded fine-tuned model weights from", model_path)
        else:
            print("⚠️ Model weights not found:", model_path)

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def run(self, input_data):
        img_path = input_data if isinstance(input_data, str) else (
            input_data.get("user_request") if isinstance(input_data, dict) else None
        )
        if img_path is None:
            return {"pathology": "", "confidence": 0.0}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return {"pathology": "ErrorOpen", "confidence": 0.0, "error": str(e)}

        imgs = [img]
        if self.tta:
            imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

        probs_list = []
        with torch.no_grad():
            for im in imgs:
                x = self.tf(im).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list.append(probs)

        avg_probs = np.mean(probs_list, axis=0)
        idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[idx])
        label = CLASS_NAMES[idx]

        if conf < self.threshold:
            label = "Other"

        return {"pathology": label, "confidence": round(conf, 4)}
