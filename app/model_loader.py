# app/model_loader.py

import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification, ViTImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# CHAMPION: FaceForge XceptionNet  (90% on FF++ C23)
# ===================================================================

class FaceForgeDetector(nn.Module):
    """XceptionNet backbone + custom head. Labels: 0=Real, 1=Fake."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=False, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def _load_champion():
    weight_path = hf_hub_download(
        repo_id="huzaifanasirrr/faceforge-detector",
        filename="detector_best.pth",
    )
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = FaceForgeDetector()
    backbone_w, classifier_w = {}, {}
    for k, v in state_dict.items():
        if k.startswith("xception."):
            backbone_w[k.replace("xception.", "", 1)] = v
        elif k.startswith("classifier."):
            classifier_w[k.replace("classifier.", "", 1)] = v

    model.backbone.load_state_dict(backbone_w, strict=False)
    model.classifier.load_state_dict(classifier_w)
    model.to(device).eval()
    print("✅ Champion loaded: FaceForge XceptionNet (90% FF++ C23)")
    return model


# ===================================================================
# CHALLENGER: ViT
# Tries to load fine-tuned model from models/vit_finetuned_ffpp_v2/
# Falls back to prithivMLmods/Deep-Fake-Detector-v2-Model
# ===================================================================

import os
from transformers import AutoModelForImageClassification, AutoImageProcessor
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CHALLENGER_MODEL_PATH = os.path.join(MODELS_DIR, "vit_finetuned_twostage")

def _load_challenger():
    m = None
    p = None
    if os.path.exists(CHALLENGER_MODEL_PATH):
        try:
            # from transformers import ViTForImageClassification, ViTImageProcessor # Already imported globally
            m = ViTForImageClassification.from_pretrained(CHALLENGER_MODEL_PATH).to(device)
            m.eval()
            p = ViTImageProcessor.from_pretrained(CHALLENGER_MODEL_PATH)
            print(f"✅ Challenger loaded: Fine-tuned ViT from {CHALLENGER_MODEL_PATH}")
            return m, p
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned ViT from {CHALLENGER_MODEL_PATH}: {e}")
            print("   Falling back to HuggingFace pretrained model.")
    
    # Fall back to HuggingFace pretrained if local load failed or directory doesn't exist
    # from transformers import ViTForImageClassification, ViTImageProcessor # Already imported globally
    name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    m = ViTForImageClassification.from_pretrained(name).to(device).eval()
    p = ViTImageProcessor.from_pretrained(name)
    print("✅ Challenger loaded: prithivMLmods ViT (HuggingFace, 52.5%)")
    print("   ℹ️  To improve: run colab_finetune_vit.py on GPU, save to models/vit_finetuned_ffpp_v2/")
    return m, p


# ===================================================================
# FALLBACK: EfficientNet
# Tries to load fine-tuned model from models/efficientnet_finetuned_ffpp/
# returning None if not found so the API can dynamically omit it.
# ===================================================================

FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnet_finetuned_ffpp")

def _load_fallback():
    m = None
    p = None
    if os.path.exists(FALLBACK_MODEL_PATH):
        try:
            m = AutoModelForImageClassification.from_pretrained(FALLBACK_MODEL_PATH).to(device)
            m.eval()
            p = AutoImageProcessor.from_pretrained(FALLBACK_MODEL_PATH)
            print(f"✅ Fallback loaded: Fine-tuned EfficientNet from {FALLBACK_MODEL_PATH}")
            return m, p
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned EfficientNet from {FALLBACK_MODEL_PATH}: {e}")
    else:
        print(f"ℹ️ Fallback model not found at {FALLBACK_MODEL_PATH}. Running 2-model ensemble.")
    
    return None, None


# Preprocessing for champion
from torchvision import transforms

champion_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# Load at startup
champion = _load_champion()
challenger_model, challenger_processor = _load_challenger()
fallback_model, fallback_processor = _load_fallback()