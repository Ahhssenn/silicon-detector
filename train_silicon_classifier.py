import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Configuration
DATA_ROOT = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MODEL_OUT = "silicon_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders():
    """Prepare dataloaders for training and validation."""

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Dataset folders not found. Expected structure:\n"
            f"{DATA_ROOT}/train/silicon\n"
            f"{DATA_ROOT}/train/non_silicon\n"
            f"{DATA_ROOT}/val/silicon\n"
            f"{DATA_ROOT}/val/non_silicon"
        )

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes=2):
    """Load a pretrained ResNet-18 and replace the classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train():
    train_loader, val_loader, classes = get_dataloaders()
    print("Detected classes:", classes)

    model = build_model(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.2f} | "
            f"Val Acc: {val_acc:.2f}"
        )

    # Save checkpoint
    torch.save(
        {"model_state": model.state_dict(), "classes": classes},
        MODEL_OUT
    )
    print(f"\nModel saved to: {MODEL_OUT}")


if __name__ == "__main__":
    train()
