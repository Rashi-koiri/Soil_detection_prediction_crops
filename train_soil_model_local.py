import os
import hashlib
import json
import time
from collections import defaultdict
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

CLASS_NAMES = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

# Custom PyTorch Dataset
class SoilImageDataset(Dataset):
    def __init__(self, file_tuples, img_size=(128, 128), augment=False):
        self.file_tuples = file_tuples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        path, label = self.file_tuples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB').resize(self.img_size)
            
            if self.augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # Random slight rotation
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    img = img.rotate(angle)

            arr = np.array(img, dtype=np.float32) / 255.0 # (H, W, C)
            # PyTorch expects (C, H, W)
            tensor = torch.tensor(arr).permute(2, 0, 1)
            # Normalize with standard mean/std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor, torch.tensor(label, dtype=torch.long)

# Clean, robust CNN Architecture
class SoilCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SoilCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def collect_unique_dataset(dataset_dir):
    print("Collecting and de-duplicating images from:", dataset_dir)
    unique_hashes = {}
    class_files = defaultdict(list)

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(root, f)
                # Check category
                cat = None
                for c in CLASS_NAMES:
                    if c in full_path:
                        cat = c
                        break
                if cat is None:
                    continue

                with open(full_path, 'rb') as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                if h not in unique_hashes:
                    unique_hashes[h] = full_path
                    class_files[cat].append(full_path)

    train_set, val_set, test_set = [], [], []

    for cat, paths in class_files.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        label = CLASS_TO_IDX[cat]

        for p in paths[:n_train]:
            train_set.append((p, label))
        for p in paths[n_train:n_train+n_val]:
            val_set.append((p, label))
        for p in paths[n_train+n_val:]:
            test_set.append((p, label))

    print(f"Total Unique Images: {len(unique_hashes)}")
    print(f"Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    return train_set, val_set, test_set

def main():
    raw_dataset = r"C:\Users\RASHI KOIRI\Downloads\archive (1)\Dataset"
    output_dir = r"C:\Users\RASHI KOIRI\OneDrive\Desktop\soil_detection"
    os.makedirs(output_dir, exist_ok=True)

    train_data, val_data, test_data = collect_unique_dataset(raw_dataset)

    train_loader = DataLoader(SoilImageDataset(train_data, augment=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(SoilImageDataset(val_data, augment=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(SoilImageDataset(test_data, augment=False), batch_size=32, shuffle=False)

    model = SoilCNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 15
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "soil_classifier_model.pth")

    print(f"\nStarting Soil CNN Training for {epochs} epochs on local CPU...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved best model checkpoint to: {best_model_path}")

    # Evaluate on Unseen Test Vault
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_correct = 0
    test_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_acc = (test_correct / test_total) * 100
    print(f"\n🎯 Final Test Accuracy on Unseen Test Set: {test_acc:.2f}%")

    # Save metadata
    meta = {
        "architecture": "SoilCNN-4Block",
        "classes": CLASS_NAMES,
        "input_size": [128, 128, 3],
        "test_accuracy": round(test_acc, 2),
        "total_unique_images": len(train_data) + len(val_data) + len(test_data),
        "best_val_accuracy": round(best_val_acc, 2)
    }
    meta_path = os.path.join(output_dir, "soil_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")

if __name__ == "__main__":
    main()
