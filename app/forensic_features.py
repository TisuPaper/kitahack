# app/forensic_features.py
"""
Image forensic feature analysis for deepfake detection.
Provides architecture-independent signals that complement the DL model.

Features computed:
1. FFT Frequency Analysis — deepfakes often lack high-frequency details
2. Noise Pattern Analysis — inconsistent noise across face regions
3. ELA (Error Level Analysis) — JPEG compression artifact inconsistencies
4. Laplacian Variance — sharpness / blurriness detection
"""

import cv2
import numpy as np
from PIL import Image


def fft_high_freq_ratio(pil_img: Image.Image) -> dict:
    """
    Compute the ratio of high-frequency energy to total energy in the FFT.
    Deepfakes often have LESS high-frequency content than real images
    because generators struggle to produce fine details.

    Returns:
        high_freq_ratio: float (0-1). Lower ≈ more suspicious.
        mean_magnitude: float. Overall frequency energy.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # 2D FFT, shift zero-frequency to center
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Create circular mask: center = low freq, edges = high freq
    cy, cx = h // 2, w // 2
    # "Low frequency" = inner 30% of the spectrum
    radius = int(min(h, w) * 0.15)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    total_energy = np.sum(magnitude ** 2)
    low_mask = dist <= radius
    low_energy = np.sum(magnitude[low_mask] ** 2)
    high_energy = total_energy - low_energy

    ratio = float(high_energy / max(total_energy, 1e-10))
    return {
        "high_freq_ratio": round(ratio, 4),
        "mean_magnitude": round(float(np.mean(np.log1p(magnitude))), 4),
    }


def noise_analysis(pil_img: Image.Image) -> dict:
    """
    Extract noise residual and analyze its consistency.
    Real images have uniform sensor noise; deepfakes may have
    inconsistent noise between the generated face and background.

    Returns:
        noise_std: float. Standard deviation of noise residual.
        noise_uniformity: float (0-1). How uniform noise is across patches.
            Lower uniformity = more suspicious (inconsistent noise).
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)

    # Extract noise residual: original - denoised
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - denoised

    noise_std = float(np.std(noise))

    # Divide into 4x4 grid and check noise consistency
    h, w = noise.shape
    ph, pw = h // 4, w // 4
    patch_stds = []
    for i in range(4):
        for j in range(4):
            patch = noise[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            if patch.size > 0:
                patch_stds.append(float(np.std(patch)))

    if len(patch_stds) > 1:
        # Coefficient of variation: lower = more uniform = more likely real
        mean_std = np.mean(patch_stds)
        cv = float(np.std(patch_stds) / max(mean_std, 1e-10))
        uniformity = max(0.0, 1.0 - cv)
    else:
        uniformity = 0.5

    return {
        "noise_std": round(noise_std, 4),
        "noise_uniformity": round(uniformity, 4),
    }


def error_level_analysis(pil_img: Image.Image, quality: int = 90) -> dict:
    """
    Error Level Analysis (ELA).
    Re-compress at a known quality and measure the difference.
    Manipulated regions show different error levels than untouched areas.

    Returns:
        ela_mean: float. Mean ELA value.
        ela_std: float. Std of ELA values.
        ela_max_deviation: float. Max deviation from mean (high = suspicious).
    """
    import io

    # Re-compress as JPEG
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    # Compute absolute difference
    orig_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    diff = np.abs(orig_arr - recomp_arr)

    # Analyze per-pixel ELA
    ela_gray = np.mean(diff, axis=2)  # average across channels
    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))

    # Patch-based deviation analysis (4x4 grid)
    h, w = ela_gray.shape
    ph, pw = h // 4, w // 4
    patch_means = []
    for i in range(4):
        for j in range(4):
            patch = ela_gray[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            if patch.size > 0:
                patch_means.append(float(np.mean(patch)))

    # Max deviation from overall mean
    if patch_means:
        max_dev = max(abs(p - ela_mean) for p in patch_means)
    else:
        max_dev = 0.0

    return {
        "ela_mean": round(ela_mean, 4),
        "ela_std": round(ela_std, 4),
        "ela_max_deviation": round(max_dev, 4),
    }


def laplacian_variance(pil_img: Image.Image) -> dict:
    """
    Variance of the Laplacian — a simple blur detection metric.
    Very sharp or very blurry images can indicate manipulation.

    Returns:
        laplacian_var: float. Higher = sharper image.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "laplacian_variance": round(float(np.var(lap)), 4),
    }


def compute_all_features(pil_img: Image.Image) -> dict:
    """Compute all forensic features for an image."""
    features = {}
    features.update(fft_high_freq_ratio(pil_img))
    features.update(noise_analysis(pil_img))
    features.update(error_level_analysis(pil_img))
    features.update(laplacian_variance(pil_img))
    return features
