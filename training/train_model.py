import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# =======================================
# Dataset
# =======================================
class SimpleXRDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


# =======================================
# Paths
# =======================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "pneumonia_dataset",
    "chest_xray",
    "train"
)

MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "chest_classifier.pt"
)

# =======================================
# Load dataset
# =======================================
classes = ["NORMAL", "PNEUMONIA"]
files, labels = [], []

for idx, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    for f in os.listdir(cls_path):
        files.append(os.path.join(cls_path, f))
        labels.append(idx)

print(f"Found {len(files)} images.")

train_files, val_files, train_labels, val_labels = train_test_split(
    files, labels, test_size=0.2, random_state=42, stratify=labels
)

# =======================================
# Transforms
# =======================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = SimpleXRDataset(train_files, train_labels, transform)
val_ds = SimpleXRDataset(val_files, val_labels, transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# =======================================
# Model
# =======================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_val_loss = float("inf")

# =======================================
# Training
# =======================================
for epoch in range(1, 6):
    print(f"Epoch {epoch}/5")

    # Training
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"  Train loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    avg_val = val_loss / len(val_loader)
    acc = correct / total * 100

    print(f"  Val Loss: {avg_val:.4f} | Acc: {acc:.2f}%")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  Saved best model → {MODEL_SAVE_PATH}\n")
