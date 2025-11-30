# agents/image_analysis_agent.py
import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["Normal", "Pneumonia", "Other"]

class ImageAnalysisAgent:
    def __init__(self, model_path="models/chest_classifier.pt", device=None,
                 threshold=0.50, tta=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tta = tta

        # modern weights API
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)

        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state)
                print("Loaded model weights from", model_path)
            except Exception as e:
                print("Warning: failed to load model weights:", e)

        # standard transform (used after preprocessing)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _clahe(self, img_gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img_gray)

    def _load_and_preprocess(self, path):
        # load grayscale via OpenCV for CLAHE, return HxWx3 uint8
        img_bgr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None:
            raise IOError(f"Cannot open {path}")
        img_eq = self._clahe(img_bgr)
        # convert to 3-channel
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
        return img_rgb

    def _tta_transforms(self, img_np):
        # yields tensors for different TTA variants
        tensors = []
        pil_base = Image.fromarray(img_np)
        base = self.tf(np.array(pil_base))
        tensors.append(base)
        if self.tta:
            # horizontal flip
            tensors.append(self.tf(np.array(pil_base.transpose(Image.FLIP_LEFT_RIGHT))))
            # small rotations
            for angle in (-10, 10):
                tensors.append(self.tf(np.array(pil_base.rotate(angle))))
        return tensors

    def _predict_probs(self, path):
        img_np = self._load_and_preprocess(path)
        t_list = self._tta_transforms(img_np)
        probs_acc = None
        self.model.eval()
        with torch.no_grad():
            for t in t_list:
                x = t.unsqueeze(0).to(self.device)
                logits = self.model(x)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if probs_acc is None:
                    probs_acc = p
                else:
                    probs_acc += p
        probs_avg = probs_acc / len(t_list)
        return probs_avg

    def run(self, input_data):
        path = input_data if isinstance(input_data, str) else (
            input_data.get("user_request") if isinstance(input_data, dict) else None)
        if path is None:
            return {"pathology":"", "confidence":"0.00"}
        try:
            probs = self._predict_probs(path)
        except Exception as e:
            return {"pathology":"ErrorOpen","confidence":"0.00","error":str(e)}

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = CLASS_NAMES[top_idx]

        # thresholding: if under threshold -> Other
        if top_prob < self.threshold:
            label = "Other"

        # also provide top3 for debugging (optional)
        top3_idx = np.argsort(-probs)[:3]
        top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]

        return {"pathology": label, "confidence": f"{top_prob:.2f}", "top3": top3}
