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
# Tries to load fine-tuned model from models/vit_finetuned_ffpp/
# Falls back to prithivMLmods/Deep-Fake-Detector-v2-Model
# ===================================================================

import os as _os
_FINETUNED_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), "models", "vit_finetuned_ffpp"
)


def _load_challenger():
    if _os.path.isdir(_FINETUNED_PATH) and _os.path.exists(
        _os.path.join(_FINETUNED_PATH, "config.json")
    ):
        # Use our GPU-fine-tuned model
        m = ViTForImageClassification.from_pretrained(_FINETUNED_PATH).to(device).eval()
        p = ViTImageProcessor.from_pretrained(_FINETUNED_PATH)
        print(f"✅ Challenger loaded: Fine-tuned ViT from {_FINETUNED_PATH}")
    else:
        # Fall back to HuggingFace pretrained
        name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        m = ViTForImageClassification.from_pretrained(name).to(device).eval()
        p = ViTImageProcessor.from_pretrained(name)
        print("✅ Challenger loaded: prithivMLmods ViT (HuggingFace, 52.5%)")
        print("   ℹ️  To improve: run colab_finetune_vit.py on GPU, save to models/vit_finetuned_ffpp/")
    return m, p


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