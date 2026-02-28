# =============================================================================
# 🔥 Deepfake ViT Challenger — TWO-STAGE Fine-Tuning Script
# =============================================================================
#
# STRATEGY:
# 1. Start with our already fine-tuned FF++ ViT weights
# 2. Lightly fine-tune (low LR, few epochs) on the 140k Real/Fake dataset
#    (which acts as our "wild" / internet-scraped proxy since WildDeepfake is offline).
#
# WHY:
# If you train only on Wild data, the model might forget FF++ artifacts.
# By doing two-stage training, the model retains its FF++ knowledge while
# learning the new diverse characteristics of the secondary dataset.
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

# ---- STEP 2: Download Secondary Dataset (140k Real/Fake) ----
print("📥 Downloading Secondary Dataset from Kaggle...")
import kagglehub
dataset_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
DATASET = os.path.join(dataset_path, "real_vs_fake", "real-vs-fake", "train")
print(f"✅ Dataset at: {DATASET}")

# ---- STEP 3: Configuration ----
SEED = 42
BATCH_SIZE = 32          
EPOCHS = 5               # 🟡 TWO-STAGE: Very few epochs! We just want to lightly adjust weights.
LR = 5e-6                # 🟡 TWO-STAGE: Very small learning rate! Don't destroy FF++ weights.
WEIGHT_DECAY = 0.05      # Higher regularization to prevent catastrophic forgetting
LABEL_SMOOTHING = 0.1    
MAX_IMAGES_PER_CLASS = 10000  # Large batch to see variety quickly
VAL_SPLIT = 0.2

# 🟡 TWO-STAGE: Start from YOUR fine-tuned FF++ model!
# Upload your vit_finetuned_ffpp folder to your Google Drive or Colab first.
MODEL_NAME = "/content/vit_finetuned_ffpp" 
SAVE_DIR = "/content/vit_finetuned_twostage"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")

# ---- STEP 4: Direct image loading ----
def load_image_paths(directory, max_images):
    images = []
    if os.path.exists(directory):
        files = os.listdir(directory)
        random.shuffle(files) 
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
if not os.path.exists(MODEL_NAME):
    print(f"\n❌ ERROR: Could not find your pre-trained FF++ model at '{MODEL_NAME}'")
    print("   To do Two-Stage training, you must:")
    print("   1. Click the 📁 Files panel on the left side of Colab")
    print("   2. Upload the `vit_finetuned_ffpp` folder you trained earlier")
    print("   3. Make sure it is located at `/content/vit_finetuned_ffpp`")
    print("   (Or point MODEL_NAME to the base 'prithivMLmods/Deep-Fake-Detector-v2-Model' to train from scratch)")
    import sys
    sys.exit(1)

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
print(f"\n🧠 Loading ALREADY-TUNED ViT from {MODEL_NAME}...")
try:
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print("\n❌ ERROR: You must upload your `vit_finetuned_ffpp` folder to Colab first!")
    print(f"The script is looking for it at: {MODEL_NAME}")
    import sys
    sys.exit(1)

# 🟡 TWO-STAGE: Only unfreeze the very top layers to prevent destroying earlier weights
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# Only unfreeze the final 2 blocks (instead of 6) to keep the core FF++ knowledge intact
for block in model.vit.encoder.layer[-2:]:
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

# No warmup needed since we are already fine-tuned, just decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
classifier_dropout = nn.Dropout(0.3)

print(f"\n{'='*60}")
print(f"🚀 Light TWO-STAGE Tuning for {EPOCHS} epochs on {device}...")
print(f"{'='*60}\n")

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Stricter clipping
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
    lr_now = scheduler.get_last_lr()[0]  

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)

    print(
        f"Epoch {epoch+1:2d}/{EPOCHS} | "
        f"loss={train_loss/len(train_dl):.4f} train={train_acc:.1f}% | "
        f"val={val_acc:.1f}% FP={val_fp} FN={val_fn} | "
        f"lr={lr_now:.2e}"
    )

print(f"\n✅ Two-stage tuning complete!")
print(f"💾 Model saved to: {SAVE_DIR}")

# ---- STEP 9: Auto-download ----
print("\n📥 Preparing download...")
try:
    import shutil
    from google.colab import files
    shutil.make_archive('/content/vit_finetuned_twostage', 'zip', SAVE_DIR)
    print("✅ Downloading model zip...")
    files.download('/content/vit_finetuned_twostage.zip')
except:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎉 DONE! Download your Two-Stage Master ViT:           ║
║                                                          ║
║  1. In Colab, click the 📁 Files panel on the left       ║
║  2. Navigate to /content/vit_finetuned_twostage/         ║
║  3. Download ALL files in that folder                    ║
╚══════════════════════════════════════════════════════════╝
""")
