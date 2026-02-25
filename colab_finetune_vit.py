# =============================================================================
# 🔥 Deepfake ViT Challenger — GPU Fine-Tuning Script for Google Colab
# =============================================================================
#
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. File → Upload Notebook → Upload this file (or paste into a cell)
# 3. Runtime → Change runtime type → GPU (T4 is fine)
# 4. Run All
# 5. Download the model from /content/vit_finetuned_ffpp/ when done
# 6. Copy downloaded folder to: kitahack/models/vit_finetuned_ffpp/
#
# Expected time: ~30-60 minutes on T4 GPU
# Expected accuracy: 75-85% on FF++ C23
# =============================================================================

# ---- STEP 1: Install dependencies ----
# !pip install -q torch torchvision transformers pillow opencv-python kagglehub

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# ---- STEP 2: Download FF++ C23 dataset ----
print("📥 Downloading FF++ C23 dataset...")
import kagglehub
dataset_path = kagglehub.dataset_download("xdxd003/ff-c23")
DATASET = os.path.join(dataset_path, "FaceForensics++_C23")
print(f"✅ Dataset at: {DATASET}")

# ---- STEP 3: Configuration ----
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_VIDEOS_PER_CLASS = 50  # Use all 50 videos per class
FRAMES_PER_VIDEO = 10      # Extract 10 frames per video
VAL_SPLIT = 0.2            # 20% for validation
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
SAVE_DIR = "/content/vit_finetuned_ffpp"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

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
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    videos = videos[:max_videos]

    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(video_dir, v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        # Skip first/last 10% (intro/outro)
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

        if (vi + 1) % 10 == 0:
            print(f"  Processed {vi + 1}/{len(videos)} videos, {len(frames)} frames")

    return frames

print("\n📂 Extracting frames...")
real_dir = os.path.join(DATASET, "original")

# Real frames
real_frames = extract_frames_from_dir(real_dir, MAX_VIDEOS_PER_CLASS, FRAMES_PER_VIDEO)
print(f"  ✅ Real: {len(real_frames)} frames")

# Fake frames from all 4 manipulation types
fake_dirs = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
fake_frames = []
for fd in fake_dirs:
    d = os.path.join(DATASET, fd)
    if os.path.exists(d):
        n_per_type = MAX_VIDEOS_PER_CLASS // len(fake_dirs)  # Balance across types
        ff = extract_frames_from_dir(d, n_per_type, FRAMES_PER_VIDEO)
        fake_frames.extend(ff)
        print(f"  ✅ {fd}: {len(ff)} frames (total fake: {len(fake_frames)})")

# Balance classes
n = min(len(real_frames), len(fake_frames))
real_frames = real_frames[:n]
fake_frames = fake_frames[:n]
print(f"\n📊 Balanced: {n} real + {n} fake = {2*n} total frames")

# ---- STEP 6: Train/Val split ----
all_frames = real_frames + fake_frames
all_labels = [0] * len(real_frames) + [1] * len(fake_frames)  # 0=Real, 1=Fake

combined = list(zip(all_frames, all_labels))
random.shuffle(combined)

split = int(len(combined) * (1 - VAL_SPLIT))
train_data = combined[:split]
val_data = combined[split:]

train_frames, train_labels = zip(*train_data)
val_frames, val_labels = zip(*val_data)
print(f"📊 Train: {len(train_frames)}, Val: {len(val_frames)}")

# ---- STEP 7: Dataset with augmentation ----
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Training augmentations to improve generalization
train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    # Simulate different compression levels
    transforms.RandomChoice([
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.Lambda(lambda x: x),  # no-op
    ]),
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

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---- STEP 8: Model setup ----
print("\n🧠 Loading ViT model...")
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

# Set correct labels
model.config.id2label = {0: "Real", 1: "Fake"}
model.config.label2id = {"Real": 0, "Fake": 1}
model.config.num_labels = 2

# Replace classifier head
model.classifier = nn.Linear(model.config.hidden_size, 2)

# Unfreeze strategy: classifier + last 4 transformer blocks
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Unfreeze last 4 encoder blocks (out of 12)
for block in model.vit.encoder.layer[-4:]:
    for param in block.parameters():
        param.requires_grad = True

# Unfreeze layer norm
for param in model.vit.layernorm.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"📦 Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# ---- STEP 9: Training loop ----
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"\n{'='*60}")
print(f"🚀 Training for {EPOCHS} epochs on {device}...")
print(f"{'='*60}\n")

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 5  # Stop after 5 epochs without improvement

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, (pixels, labels) in enumerate(train_dl):
        pixels = pixels.to(device)
        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
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
    lr_now = scheduler.get_last_lr()[0]

    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
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

# ---- STEP 10: Final benchmark ----
print("\n" + "=" * 60)
print("📊 Final Benchmark on 20 real + 20 fake videos (frame 30)")
print("=" * 60)

# Reload best model
model = ViTForImageClassification.from_pretrained(SAVE_DIR).to(device).eval()

correct_real, correct_fake, total_real, total_fake = 0, 0, 0, 0

def get_frame(video_path, frame_idx=30):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

# Test real videos
for v in sorted(os.listdir(real_dir))[:20]:
    if not v.endswith(".mp4"):
        continue
    img = get_frame(os.path.join(real_dir, v))
    if img is None:
        continue
    img = crop_face(img)
    total_real += 1
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    status = "✅" if pred == 0 else "❌"
    print(f"  {status} REAL {v}: {'Real' if pred==0 else 'Fake'} "
          f"(real={float(probs[0][0]):.3f}, fake={float(probs[0][1]):.3f})")
    if pred == 0:
        correct_real += 1

# Test fake videos
fake_test_dir = os.path.join(DATASET, "Deepfakes")
for v in sorted(os.listdir(fake_test_dir))[:20]:
    if not v.endswith(".mp4"):
        continue
    img = get_frame(os.path.join(fake_test_dir, v))
    if img is None:
        continue
    img = crop_face(img)
    total_fake += 1
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    status = "✅" if pred == 1 else "❌"
    print(f"  {status} FAKE {v}: {'Real' if pred==0 else 'Fake'} "
          f"(real={float(probs[0][0]):.3f}, fake={float(probs[0][1]):.3f})")
    if pred == 1:
        correct_fake += 1

total_correct = correct_real + correct_fake
total_all = total_real + total_fake
print(f"\n{'='*60}")
print(f"📊 RESULTS:")
print(f"   Real accuracy:  {correct_real}/{total_real} ({correct_real/max(total_real,1)*100:.0f}%)")
print(f"   Fake accuracy:  {correct_fake}/{total_fake} ({correct_fake/max(total_fake,1)*100:.0f}%)")
print(f"   Overall:        {total_correct}/{total_all} ({total_correct/max(total_all,1)*100:.0f}%)")
print(f"{'='*60}")

# ---- STEP 11: Download instructions ----
print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎉 DONE! Download your fine-tuned model:               ║
║                                                          ║
║  1. In Colab, click the 📁 Files panel on the left       ║
║  2. Navigate to /content/vit_finetuned_ffpp/             ║
║  3. Download ALL files in that folder                    ║
║  4. Place them in your project at:                       ║
║     kitahack/models/vit_finetuned_ffpp/                  ║
║  5. The model_loader.py will auto-detect them            ║
╚══════════════════════════════════════════════════════════╝
""")
