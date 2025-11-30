# agents/image_analysis_agent.py
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# adjust these classes as you train more labels
CLASS_NAMES = ["Normal", "Pneumonia", "Other"]

class ImageAnalysisAgent:
    """
    Demonstration ImageAnalysisAgent with simple TTA and tie-break heuristics
    to reduce 'Other' predictions when model is uncertain.
    """

    def __init__(self, model_path="models/chest_classifier.pt", device=None, tta=True, tie_delta=0.15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tta = tta
        self.tie_delta = tie_delta

        # build model using modern weights API if available (avoids deprecation warnings)
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        except Exception:
            # fallback for older torchvision
            backbone = models.resnet18(pretrained=True)

        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        self.model = backbone.to(self.device)
        self.model.eval()

        # try to load weights if provided
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                # Allow state to be either full model or state_dict for fc
                if isinstance(state, dict) and not any(k.startswith("fc.") for k in state.keys()):
                    # maybe it's a full model state_dict with keys matching
                    self.model.load_state_dict(state)
                else:
                    self.model.load_state_dict(state)
                print("Loaded model weights from", model_path)
            except Exception as e:
                print("Warning: failed to load model weights:", e)

        # base transform
        self.base_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _predict_probs(self, pil_img):
        """
        Return averaged probability vector (numpy) using TTA if enabled.
        """
        self.model.eval()
        probs_accum = None
        tfs = [self.base_tf]
        if self.tta:
            # additional TTA: horizontal flip
            def hflip_transform(img):
                return self.base_tf(transforms.functional.hflip(img))
            tfs.append(hflip_transform)

        with torch.no_grad():
            for tf in tfs:
                if callable(tf):
                    x = tf(pil_img)
                else:
                    x = tf(pil_img)
                x = x.unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if probs_accum is None:
                    probs_accum = probs
                else:
                    probs_accum = probs_accum + probs

        probs_avg = probs_accum / len(tfs)
        return probs_avg

    def run(self, input_data):
        """
        input_data: path string or dict with 'user_request'
        returns: {"pathology": label, "confidence": "0.95"}
        """
        img_path = input_data if isinstance(input_data, str) else (
            input_data.get("user_request") if isinstance(input_data, dict) else None
        )
        if img_path is None:
            return {"pathology": "", "confidence": "0%"}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return {"pathology": "ErrorOpen", "confidence": "0%", "error": str(e)}

        probs = self._predict_probs(img)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label = CLASS_NAMES[top_idx]

        # tie-break: if predicted 'Other' but second best is close, pick second
        if label.lower() == "other":
            sorted_idx = np.argsort(-probs)
            second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else None
            if second_idx is not None:
                second_prob = float(probs[second_idx])
                if (top_prob - second_prob) < self.tie_delta:
                    label = CLASS_NAMES[second_idx]
                    top_prob = second_prob

        return {"pathology": label, "confidence": f"{top_prob:.2f}"}
