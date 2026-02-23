# app/model_loader.py

import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceForgeDetector(nn.Module):
    """
    FaceForge Detector: XceptionNet backbone + custom classification head.
    Trained on FaceForensics++ (C40 compression), 99.33% accuracy on test set.
    Labels: 0=Real, 1=Fake
    """

    def __init__(self):
        super().__init__()
        # XceptionNet backbone (timm legacy_xception)
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=False, num_classes=0
        )
        # Custom head: Dropout(0.5) → FC(2048→512)+ReLU → Dropout(0.3) → FC(512→2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def load_model():
    """Download and load the FaceForge detector."""
    # Download weights from HuggingFace
    weight_path = hf_hub_download(
        repo_id="huzaifanasirrr/faceforge-detector",
        filename="detector_best.pth",
    )

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Build the model
    model = FaceForgeDetector()

    # Strip 'xception.' prefix from backbone weights, 'classifier.' from head
    backbone_weights = {}
    classifier_weights = {}
    for k, v in state_dict.items():
        if k.startswith("xception."):
            backbone_weights[k.replace("xception.", "", 1)] = v
        elif k.startswith("classifier."):
            classifier_weights[k.replace("classifier.", "", 1)] = v

    # Load weights
    model.backbone.load_state_dict(backbone_weights, strict=False)
    model.classifier.load_state_dict(classifier_weights)

    model.to(device)
    model.eval()
    print("✅ FaceForge Detector loaded (XceptionNet, FF++ C40, 99.33% accuracy)")
    return model


# Preprocessing: matches the model card exactly
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# Load at startup
model = load_model()