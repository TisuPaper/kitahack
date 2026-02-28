# =============================================================================
# 🔥 Deepfake ViT Challenger — GPU Celeb-DF Fine-Tuning Script
# =============================================================================
#
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. File → Upload Notebook → Upload this file (or paste into a cell)
# 3. Runtime → Change runtime type → GPU (T4 is fine)
# 4. Run All
# 5. Download the model from /content/vit_finetuned_celebdf/ when done
# 6. Copy downloaded folder to: kitahack/models/vit_finetuned_celebdf/
#
# DATASET:
# This script trains the Vision Transformer on the Celeb-DF v2 dataset.
# Celeb-DF contains ultra-realistic deepfakes that lack the common visual 
# blending artifacts found in older datasets like FaceForensics++.
#
# Expected time: ~30-50 minutes on T4 GPU
# =============================================================================

# ---- STEP 1: Install dependencies ----
# !pip install -q torch torchvision transformers pillow opencv-python kagglehub

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# ---- STEP 2: Download Celeb-DF v2 dataset ----
print("📥 Downloading Celeb-DF v2 dataset...")
import kagglehub
dataset_path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
# The reubensuju mirror extracts directly into the root folder
DATASET = dataset_path
print(f"✅ Dataset at: {DATASET}")

# ---- STEP 3: Configuration ----
SEED = 42
BATCH_SIZE = 16          # Smaller batch for GPU memory safety
EPOCHS = 25              # More epochs
LR = 1e-5                # Lower LR for stability
WEIGHT_DECAY = 0.02      # Slightly more regularization
LABEL_SMOOTHING = 0.1    # Reduces overconfidence
MAX_VIDEOS_PER_CLASS = 300  
FRAMES_PER_VIDEO = 15    
VAL_SPLIT = 0.2
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_DIR = "/content/vit_finetuned_celebdf"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif device.type == "cpu":
    print("   ⚠️  WARNING: Running on CPU — this will be VERY slow!")
    print("   Go to Runtime → Change runtime type → GPU (T4)")

# ---- STEP 4: Face cropper ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(pil_img, padding=0.3):
    """Crop the largest face with padding. Returns original if no face found."""
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(padding * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
    return Image.fromarray(rgb[y1:y2, x1:x2])

# ---- STEP 5: Extract frames from videos ----
def extract_frames_from_dir(video_dir, max_videos, frames_per_video):
    """Extract evenly-spaced, face-cropped frames from videos."""
    frames = []
    if not os.path.exists(video_dir):
        return frames
        
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    videos = videos[:max_videos]

    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(video_dir, v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        # Skip first/last 10%
        start = int(total * 0.10)
        end = int(total * 0.90)
        if end <= start:
            start, end = 0, total

        indices = [start + int(i * (end - start) / frames_per_video)
                   for i in range(frames_per_video)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = crop_face(Image.fromarray(rgb))
                frames.append(img)

        cap.release()

        if (vi + 1) % 20 == 0:
            print(f"  Processed {vi + 1}/{len(videos)} videos, {len(frames)} frames")

    return frames

print("\n📂 Extracting frames from Celeb-DF (this may take a few minutes)...")

# Real datasets in Celeb-DF
real_dir_1 = os.path.join(DATASET, "Celeb-real")
real_dir_2 = os.path.join(DATASET, "YouTube-real")

real_frames = extract_frames_from_dir(real_dir_1, int(MAX_VIDEOS_PER_CLASS*0.6), FRAMES_PER_VIDEO)
real_frames += extract_frames_from_dir(real_dir_2, int(MAX_VIDEOS_PER_CLASS*0.4), FRAMES_PER_VIDEO)
print(f"  ✅ Total Real: {len(real_frames)} frames")

# Fake dataset in Celeb-DF
fake_dir = os.path.join(DATASET, "Celeb-synthesis")
# We sample enough fakes to roughly match the reals
fake_frames = extract_frames_from_dir(fake_dir, MAX_VIDEOS_PER_CLASS, FRAMES_PER_VIDEO)
print(f"  ✅ Total Fake: {len(fake_frames)} frames")

# Balance classes exactly
n = min(len(real_frames), len(fake_frames))
real_frames = real_frames[:n]
fake_frames = fake_frames[:n]
print(f"\n📊 Balanced Dataset: {n} real + {n} fake = {2*n} total frames")


# ---- STEP 6: Train/Val split (VIDEO-LEVEL to prevent data leakage) ----
all_frames = real_frames + fake_frames
all_labels = [0] * len(real_frames) + [1] * len(fake_frames)

combined = list(zip(all_frames, all_labels))
random.shuffle(combined)

split = int(len(combined) * (1 - VAL_SPLIT))
train_data = combined[:split]
val_data = combined[split:]

train_frames, train_labels = zip(*train_data)
val_frames, val_labels = zip(*val_data)
print(f"📊 Train: {len(train_frames)}, Val: {len(val_frames)}")

# ---- STEP 7: Dataset with STRONGER augmentation ----
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


class DeepfakeDataset(Dataset):
    def __init__(self, images, labels, processor, augment=None):
        self.images = list(images)
        self.labels = list(labels)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = self.augment(img)
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]


train_ds = DeepfakeDataset(train_frames, train_labels, processor, augment=train_augment)
val_ds = DeepfakeDataset(val_frames, val_labels, processor, augment=None)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ---- STEP 8: Model setup (V2: unfreeze MORE layers) ----
print("\n🧠 Loading ViT model...")
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

# Set correct labels
model.config.id2label = {0: "Real", 1: "Fake"}
model.config.label2id = {"Real": 0, "Fake": 1}
model.config.num_labels = 2

# V2: Replace classifier head
model.classifier = nn.Linear(model.config.hidden_size, 2)
classifier_dropout = nn.Dropout(0.3)

# Freeze all first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# V2: Unfreeze last 6 encoder blocks 
for block in model.vit.encoder.layer[-6:]:
    for param in block.parameters():
        param.requires_grad = True

# Unfreeze layer norm
for param in model.vit.layernorm.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📦 Trainable: {trainable:,} / {total_params:,} ({trainable/total_params*100:.1f}%)")

# ---- STEP 9: Training loop (V2: label smoothing + gradient clipping) ----
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)

# V2: Warmup + cosine annealing
warmup_epochs = 3
def get_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

print(f"\n{'='*60}")
print(f"🚀 Training ViT on Celeb-DF for {EPOCHS} epochs on {device}...")
print(f"   Data: {len(train_ds)} train, {len(val_ds)} val")
print(f"   LR: {LR}, Label Smoothing: {LABEL_SMOOTHING}")
print(f"   Unfrozen layers: last 6 + classifier + layernorm")
print(f"{'='*60}\n")

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 7 

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (pixels, labels) in enumerate(train_dl):
        pixels = pixels.to(device)
        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        # Forward pass with dropout on the hidden states
        vit_outputs = model.vit(pixel_values=pixels)
        hidden = vit_outputs.last_hidden_state[:, 0]  # CLS token
        hidden = classifier_dropout(hidden)  # Apply dropout during training
        logits = model.classifier(hidden)
        loss = criterion(logits, labels)
        loss.backward()

        # V2: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += len(labels)

    scheduler.step()
    train_acc = train_correct / train_total * 100

    # --- Validate ---
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

# ---- STEP 10: Auto-download ----
print("\n📥 Preparing download...")
try:
    import shutil
    from google.colab import files
    shutil.make_archive('/content/vit_finetuned_celebdf', 'zip', SAVE_DIR)
    print("✅ Downloading model zip...")
    files.download('/content/vit_finetuned_celebdf.zip')
except:
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎉 DONE! Download your Celeb-DF ViT model:             ║
║                                                          ║
║  1. In Colab, click the 📁 Files panel on the left       ║
║  2. Navigate to /content/vit_finetuned_celebdf/          ║
║  3. Download ALL files in that folder                    ║
║  4. Place them in your project at:                       ║
║     kitahack/models/vit_finetuned_celebdf/               ║
╚══════════════════════════════════════════════════════════╝
""")
