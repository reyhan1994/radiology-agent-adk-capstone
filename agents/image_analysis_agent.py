# agents/image_analysis_agent.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["Normal", "Pneumonia", "Other"]

class ImageAnalysisAgent:
    """
    Updated version with:
    - Fine-tuned weights (models/chest_classifier.pt)
    - Threshold for 'Other' class
    - Simple Test-Time Augmentation (TTA)
    """

    def __init__(self, model_path="models/chest_classifier.pt", threshold=0.45, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # Build model
        backbone = models.resnet18(pretrained=False)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        # Load fine-tuned weights
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print("✅ Loaded fine-tuned model weights from", model_path)
        else:
            print("⚠️ Model weights not found. Using untrained ResNet18.")

        # Transform for inference
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
            return {"pathology": "", "confidence": "0%"}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return {"pathology": "ErrorOpen", "confidence": "0%", "error": str(e)}

        # --- TTA: original + horizontal flip ---
        imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
        probs_list = []

        with torch.no_grad():
            for im in imgs:
                x = self.tf(im).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list.append(probs)

        # Average TTA probabilities
        avg_probs = np.mean(probs_list, axis=0)
        idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[idx])
        label = CLASS_NAMES[idx]

        # Apply threshold for 'Other'
        if conf < self.threshold:
            label = "Other"

        return {"pathology": label, "confidence": f"{conf:.2f}"}
