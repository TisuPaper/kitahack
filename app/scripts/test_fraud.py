#!/usr/bin/env python3
"""
Test the fraud pipeline: load app/.env, then run STT → PII filter → Gemini.
Run from repo root: python app/scripts/test_fraud.py
Uses app/test_fraud.wav (create with: see SETUP_FRAUD_DETECTION.md).
"""
import os
import sys
from pathlib import Path

# Repo root = parent of app/
repo_root = Path(__file__).resolve().parent.parent.parent
os.chdir(repo_root)

# Load app/.env and app/.env.local before any app.fraud_detection import
for name in (".env", ".env.local"):
    env_path = Path("app") / name
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'").strip()
                os.environ[k.strip()] = v  # .env.local loaded second, overrides

# Now import and run
from app.fraud_detection import run_fraud_pipeline

wav = Path("app/test_fraud.wav")
if not wav.exists():
    print("Create app/test_fraud.wav first (1 sec WAV). Exiting.")
    sys.exit(1)

print("Running fraud pipeline on", wav, "...")
result = run_fraud_pipeline(str(wav))

if result.get("error"):
    print("ERROR:", result["error"])
    sys.exit(1)

print("transcript_raw:", repr(result["transcript_raw"][:100]))
print("transcript_filtered:", repr(result["transcript_filtered"][:100]))
print("redacted_count:", result["redacted_count"])
print("analysis:", result["analysis"])
print("OK")
