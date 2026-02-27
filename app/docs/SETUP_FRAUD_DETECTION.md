# Step-by-step: Backend setup for AI fraud detection

This guide sets up the **fraud detection pipeline** in the `app/` backend: **receive audio → speech-to-text → filter PII → send to Gemini for analysis**. The same backend can be run locally or deployed on a VM.

**How it uses the existing backend:** The backend is already set up (FastAPI, CORS, `/predict`, `/predict-video`, `/predict-audio`). Fraud detection **reuses that same app**: it receives audio the same way (same endpoint pattern, same supported formats as `/predict-audio`), then runs a **different processing path** on that audio: STT → PII filter → Gemini. No separate server; just one extra endpoint and extra dependencies (Whisper, Gemini).

---

## 1. Prerequisites

- **Python 3.10+** (recommended 3.11 or 3.12)
- **ffmpeg** installed on the system (required by Whisper for audio decoding)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- A **Google AI (Gemini) API key** for the analysis step ([get one here](https://aistudio.google.com/apikey))

---

## 2. Create a virtual environment (recommended)

From the **repository root** (parent of `app/`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or:  .venv\Scripts\activate   # Windows
```

---

## 3. Install backend dependencies

Install the main backend dependencies first (FastAPI, PyTorch, etc., as your project already uses). Then install the **fraud detection** dependencies:

```bash
pip install -r app/requirements-fraud.txt
```

If your project has a single `requirements.txt` at the repo root, install that too, then add:

```bash
pip install openai-whisper google-generativeai
```

---

## 4. Set the Gemini API key

The pipeline needs a **Gemini API key** for the “analyze for fraud” step. Set it as an environment variable:

**Linux / macOS (current shell):**

```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Persist for your user (e.g. in `~/.bashrc` or `~/.zshrc`):**

```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Or use a file in the app:** create `app/.env` or `app/.env.local` with one line: `GEMINI_API_KEY=your-api-key-here`. The app loads both (`.env.local` overrides `.env`). Add `.env` and `.env.local` to `.gitignore` so the key is not committed.

**When deploying on a VM**, set `GEMINI_API_KEY` in the same way (e.g. in the systemd service file, or in a `.env` file loaded by your process manager). Never commit the key to git.

---

## 5. Optional: Configure fraud detection

| Environment variable | Purpose | Default |
|----------------------|--------|--------|
| `GEMINI_API_KEY` | Required for Gemini analysis | (none) |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `WHISPER_MODEL` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` | `base` |
| `FRAUD_USE_GEMINI_STT` | Use Gemini for transcription instead of Whisper | (off) |
| `FRAUD_PII_PATTERNS_FILE` | Path to file with extra PII regex patterns (one per line) | (none) |

Example for a smaller/faster Whisper model:

```bash
export WHISPER_MODEL="tiny"
```

Example for using Gemini for speech-to-text as well (uses same API key):

```bash
export FRAUD_USE_GEMINI_STT=1
```

---

## 6. Run the backend locally

From the **repository root**:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use your Makefile:

```bash
make run-backend
```

You should see the app start and no errors about missing modules. The **first request** to `/analyze-fraud` may be slower while Whisper loads the model.

---

## 7. Test the fraud detection endpoint

**Using curl:**

```bash
curl -X POST "http://localhost:8000/analyze-fraud" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/audio.wav"
```

**Using HTTPie:**

```bash
http -f POST http://localhost:8000/analyze-fraud file@/path/to/audio.mp3
```

**Expected response shape:**

```json
{
  "request_id": "uuid",
  "transcript_raw": "original text from speech-to-text",
  "transcript_filtered": "text after PII redaction",
  "redacted_count": 0,
  "analysis": {
    "risk_level": "low",
    "summary": "One sentence summary.",
    "indicators": [],
    "recommendation": "One sentence recommendation."
  },
  "privacy": { "stored_media": false },
  "timing_ms": { "total": 1234 }
}
```

If you get **“Speech-to-text failed”** or **“Whisper not installed”**, run `pip install openai-whisper` and ensure **ffmpeg** is on your PATH.  
If you get **“Gemini analysis failed”** or **“GEMINI_API_KEY is not set”**, check step 4.

---

## 8. Deploy on a VM

1. **Copy the backend**  
   Upload the `app/` folder (and any shared config/root files your app needs) to the VM.

2. **Install system dependencies**  
   Install Python 3.10+, ffmpeg, and (if needed) the same CUDA/runtime stack you use locally.

3. **Create a venv and install deps**  
   Same as steps 2–3: create a venv in the project root and run:
   ```bash
   pip install -r app/requirements-fraud.txt
   ```
   (plus your main backend requirements if separate.)

4. **Set environment variables**  
   Set `GEMINI_API_KEY` (and optionally `WHISPER_MODEL`, `GEMINI_MODEL`, etc.) in the environment used to start the app (e.g. systemd, supervisor, or a `.env` loaded by your runner).

5. **Run the API**  
   Example with **uvicorn** and **systemd**:

   ```ini
   [Unit]
   Description=Fraud detection API
   After=network.target

   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/path/to/kitahack
   Environment="GEMINI_API_KEY=your-key"
   ExecStart=/path/to/kitahack/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Then:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable fraud-api
   sudo systemctl start fraud-api
   ```

6. **Reverse proxy (optional)**  
   Put Nginx or Caddy in front of `http://127.0.0.1:8000` for HTTPS and rate limiting.

---

## 9. Pipeline summary

| Step | Component | Purpose |
|------|-----------|--------|
| 1 | **Receive audio** | `POST /analyze-fraud` with multipart `file` (WAV, MP3, OGG, etc.) |
| 2 | **Speech-to-text** | Whisper (default) or Gemini transcribes audio → `transcript_raw` |
| 3 | **Filter PII** | Regex-based redaction of numbers, emails, phones, keywords → `transcript_filtered`, `redacted_count` |
| 4 | **Gemini analysis** | Filtered text sent to Gemini → `analysis.risk_level`, `summary`, `indicators`, `recommendation` |

PII filtering uses built-in patterns (card numbers, SSN-like, email, phone, OTP/password, etc.). You can add more patterns in a file and set `FRAUD_PII_PATTERNS_FILE` to its path (one regex per line).

---

## 10. Troubleshooting

- **“No module named 'whisper'”**  
  Run: `pip install openai-whisper`. Ensure ffmpeg is installed.

- **“GEMINI_API_KEY is not set”**  
  Set the env var in the same shell (or process) that starts uvicorn.

- **“Unsupported format”**  
  Use one of: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`, `.aac`.

- **Slow first request**  
  Whisper loads the model on first use; later requests are faster.

- **VM out of memory**  
  Use a smaller Whisper model: `export WHISPER_MODEL=tiny` or `base`.

If you need to switch to the newer Google GenAI SDK later, the same pipeline (STT → PII filter → Gemini) can be updated to use `google-genai` instead of `google-generativeai`; the endpoint and response shape can stay the same.
