# =============================================================================
# 🔥 Deepfake ViT Challenger — GPU 140k Real/Fake Fine-Tuning Script
# =============================================================================
#
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. File → Upload Notebook → Upload this file (or paste into a cell)
# 3. Runtime → Change runtime type → GPU (T4 is fine)
# 4. Run All
# 5. Download the model from /content/vit_finetuned_140k/ when done
# 6. Copy downloaded folder to: kitahack/models/vit_finetuned_140k/
#
# DATASET:
# This script trains the Vision Transformer on the "140k Real and Fake Faces" dataset.
# Unlike FaceForensics++ (which relies on video blending) or Celeb-DF, this dataset
# contains 70,000 real photos and 70,000 StyleGAN generated faces. This teaches the ViT 
# to spot completely fabricated faces (Diffusion/GAN models) rather than just face-swaps.
#
# Expected time: ~45-60 minutes on T4 GPU
# =============================================================================

# ---- STEP 1: Install dependencies ----
# !pip install -q torch torchvision transformers pillow opencv-python kagglehub

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# ---- STEP 2: Download 140k Real/Fake dataset ----
print("📥 Downloading 140k Real/Fake dataset from Kaggle...")
import kagglehub
# This extracts into a folder structure of real_vs_fake/real-vs-fake/train/
dataset_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
DATASET = os.path.join(dataset_path, "real_vs_fake", "real-vs-fake", "train")
print(f"✅ Dataset at: {DATASET}")

# ---- STEP 3: Configuration ----
SEED = 42
BATCH_SIZE = 32          # ViT can handle 32 on T4 if it's just images (not videos)
EPOCHS = 15              # Less epochs needed because the dataset is MASSIVE
LR = 1e-5                
WEIGHT_DECAY = 0.02      
LABEL_SMOOTHING = 0.1    
MAX_IMAGES_PER_CLASS = 6000  # We sample a giant chunk to train quickly
VAL_SPLIT = 0.2
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_DIR = "/content/vit_finetuned_140k"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")

# ---- STEP 4: Direct image loading (No Face Cropper needed!) ----
# The 140k dataset is already pre-cropped faces. We save massive amounts of time.
def load_image_paths(directory, max_images):
    images = []
    if os.path.exists(directory):
        files = os.listdir(directory)
        random.shuffle(files) # Shuffle before taking the chunk
        for f in files[:max_images]:
            if f.endswith(".jpg") or f.endswith(".png"):
                images.append(os.path.join(directory, f))
    return images

print("\n📂 Loading image paths...")

real_dir = os.path.join(DATASET, "real")
fake_dir = os.path.join(DATASET, "fake")

real_paths = load_image_paths(real_dir, MAX_IMAGES_PER_CLASS)
print(f"  ✅ Real: {len(real_paths)} images")

fake_paths = load_image_paths(fake_dir, MAX_IMAGES_PER_CLASS)
print(f"  ✅ Fake: {len(fake_paths)} images")

n = min(len(real_paths), len(fake_paths))
real_paths = real_paths[:n]
fake_paths = fake_paths[:n]
print(f"\n📊 Balanced Dataset: {n} real + {n} fake = {2*n} total images")

# ---- STEP 5: Train/Val split ----
all_paths = real_paths + fake_paths
all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

combined = list(zip(all_paths, all_labels))
random.shuffle(combined)

split = int(len(combined) * (1 - VAL_SPLIT))
train_data = combined[:split]
val_data = combined[split:]

train_paths, train_labels = zip(*train_data)
val_paths, val_labels = zip(*val_data)
print(f"📊 Train: {len(train_paths)}, Val: {len(val_paths)}")

# ---- STEP 6: Dataset with augmentations ----
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), 
    transforms.RandomChoice([
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.GaussianBlur(5, sigma=(0.5, 1.5)),
        transforms.Lambda(lambda x: x),
    ]),
    transforms.RandomGrayscale(p=0.1),
])

class DeepfakeImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # We load the image on demand to save RAM
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.augment:
            img = self.augment(img)
            
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]

train_ds = DeepfakeImageDataset(train_paths, train_labels, processor, augment=train_augment)
val_ds = DeepfakeImageDataset(val_paths, val_labels, processor, augment=None)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ---- STEP 7: Model setup ----
print("\n🧠 Loading ViT model...")
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

model.config.id2label = {0: "Real", 1: "Fake"}
model.config.label2id = {"Real": 0, "Fake": 1}
model.config.num_labels = 2

model.classifier = nn.Linear(model.config.hidden_size, 2)
classifier_dropout = nn.Dropout(0.3)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

for block in model.vit.encoder.layer[-6:]:
    for param in block.parameters():
        param.requires_grad = True

for param in model.vit.layernorm.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📦 Trainable: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)")

# ---- STEP 8: Training loop ----
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)

warmup_epochs = 3
def get_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

print(f"\n{'='*60}")
print(f"🚀 Training ViT on 140k Dataset for {EPOCHS} epochs on {device}...")
print(f"{'='*60}\n")

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 5 

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (pixels, labels) in enumerate(train_dl):
        pixels = pixels.to(device)
        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        vit_outputs = model.vit(pixel_values=pixels)
        hidden = vit_outputs.last_hidden_state[:, 0]  
        hidden = classifier_dropout(hidden)  
        logits = model.classifier(hidden)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += len(labels)

    scheduler.step()
    train_acc = train_correct / train_total * 100

    model.eval()
    val_correct, val_total = 0, 0
    val_fp, val_fn = 0, 0

    with torch.no_grad():
        for pixels, labels in val_dl:
            pixels = pixels.to(device)
            labels_t = torch.tensor(labels).to(device)
            outputs = model(pixel_values=pixels)
            preds = torch.argmax(outputs.logits, dim=1)

            for p, l in zip(preds.tolist(), labels):
                val_total += 1
                if p == l:
                    val_correct += 1
                elif p == 1 and l == 0:
                    val_fp += 1
                else:
                    val_fn += 1

    val_acc = val_correct / max(val_total, 1) * 100
    lr_now = scheduler.get_last_lr()[0] * LR 

    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        improved = " ⭐ best!"
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch+1:2d}/{EPOCHS} | "
        f"loss={train_loss/len(train_dl):.4f} train={train_acc:.1f}% | "
        f"val={val_acc:.1f}% FP={val_fp} FN={val_fn} | "
        f"lr={lr_now:.2e}{improved}"
    )

    if patience_counter >= PATIENCE:
        print(f"\n⏹ Early stopping — no improvement for {PATIENCE} epochs")
        break

print(f"\n✅ Best validation accuracy: {best_val_acc:.1f}%")
print(f"💾 Model saved to: {SAVE_DIR}")

# ---- STEP 9: Auto-download ----
print("\n📥 Preparing download...")
try:
    import shutil
    from google.colab import files
    shutil.make_archive('/content/vit_finetuned_140k', 'zip', SAVE_DIR)
    print("✅ Downloading model zip...")
    files.download('/content/vit_finetuned_140k.zip')
except:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎉 DONE! Download your 140K GAN ViT model:             ║
║                                                          ║
║  1. In Colab, click the 📁 Files panel on the left       ║
║  2. Navigate to /content/vit_finetuned_140k/             ║
║  3. Download ALL files in that folder                    ║
║  4. Place them in your project at:                       ║
║     kitahack/models/vit_finetuned_140k/                  ║
╚══════════════════════════════════════════════════════════╝
""")
