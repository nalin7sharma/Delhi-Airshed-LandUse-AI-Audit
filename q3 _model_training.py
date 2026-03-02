import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# CONFIG
IMAGE_FOLDER = "Delhi-Airshed-LandUse-AI-Audit\\rgb"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Train/Test CSV
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

# Encode labels
classes = train_df["label"].unique()
class_to_idx = {label: idx for idx, label in enumerate(classes)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

train_df["label_idx"] = train_df["label"].map(class_to_idx)
test_df["label_idx"] = test_df["label"].map(class_to_idx)


# Custom Dataset
class LandUseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["filename"]
        label = self.df.iloc[idx]["label_idx"]

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = LandUseDataset(train_df, transform)
test_dataset = LandUseDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load ResNet18

model = models.resnet18(pretrained=True)

# Modify final layer
num_classes = len(classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

print("Training Complete!")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)

# F1 Score
f1 = f1_score(all_labels, all_preds, average="weighted")

print("Accuracy:", acc)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")