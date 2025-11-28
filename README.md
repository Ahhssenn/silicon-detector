# Silicon Detector – Image Classification Model

This project provides a lightweight PyTorch model for classifying field images as either **silicon applied** or **no silicon**.
It is designed for real inspection workflows such as RMU panels, meter boxes, and utility field documentation.

---

## 🔍 Overview

Field engineers frequently submit photos as part of maintenance or installation tasks. Determining whether silicon sealing is applied correctly is usually done manually and is often slow or inconsistent.

This model automates that step by performing binary image classification:

- `silicon`
- `non_silicon`

The model is based on **ResNet-18**, fine-tuned on a custom dataset of labeled images. Validation accuracy typically reaches **~95–96%**.

---

## 📁 Dataset Structure

Your dataset must follow this folder structure:

```
dataset/
├── train/
│   ├── silicon/
│   └── non_silicon/
│
└── val/
    ├── silicon/
    └── non_silicon/
```

Each folder contains regular `.jpg` or `.png` images.

---

## ⚙️ Training Script

The project includes a training pipeline (`train_silicon_classifier.py`) that:

- Loads the dataset in the structure above
- Applies augmentations (flip, rotation, normalization)
- Fine-tunes a pretrained **ResNet-18**
- Computes accuracy after each epoch
- Saves the trained model as `silicon_model.pth`

---

## 🚀 How to Train

Install dependencies:

```bash
pip install torch torchvision

python train_silicon_classifier.py
```

Expected output:

```
Epoch 10/10 | Train Acc: 1.00 | Val Acc: 0.96
✔ Saved trained model as: silicon_model.pth
```

The output model file `silicon_model.pth` contains:

- The trained model weights
- The class labels used during training

You can load it in a separate script to run inference.

---

## 🧠 How the Model Works

- **Architecture:** ResNet-18 (ImageNet pretrained)
- **Final layer:** Replaced with `Linear(in_features → 2)`
- **Optimizer:** Adam (learning rate = 1e-4)
- **Loss:** CrossEntropyLoss
- **Inputs:** 256×256 RGB images, normalized with ImageNet statistics

---

## 📗 How to Use the Model (Inference)

The project includes a helper class `SiliconClassifier` in `model.py` for easy inference:

### Using the SiliconClassifier Class

```python
from model import SiliconClassifier

# Initialize the classifier
clf = SiliconClassifier(model_path="silicon_model.pth")

# Get prediction and confidence
label, confidence = clf.predict("path/to/image.jpg")
print(f"Label: {label}, Confidence: {confidence:.3f}")

# Check if silicon with threshold
is_silicon = clf.is_silicon("path/to/image.jpg", silicon_label="silicon")
print(f"Is silicon: {is_silicon}")
```

### Command-Line Usage

You can also use the `infer.py` script:

```bash
python infer.py --image path/to/image.jpg --model silicon_model.pth --threshold 0.5
```

Output:

```
Image: path/to/image.jpg
Model: silicon_model.pth
Prediction: silicon
Confidence: 0.956
Is silicon: True
```

### Manual Model Loading

If you prefer to load the model directly:

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load the saved model
checkpoint = torch.load("silicon_model.pth", map_location="cpu")
classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Image transform
tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    prob = torch.softmax(y, dim=1)[0]
    idx = int(prob.argmax())
    return classes[idx], float(prob[idx])

label, confidence = predict("sample.jpg")
print(label, confidence)
```

---

## 📦 Project Files

- **`train_silicon_classifier.py`** - Main training script
- **`model.py`** - `SiliconClassifier` helper class for inference
- **`infer.py`** - Command-line inference script
- **`README.md`** - This file
- **`LICENSE`** - MIT License

---

## 📄 License

This project is released under the MIT License.  
You may use it for personal, commercial, and academic applications.
