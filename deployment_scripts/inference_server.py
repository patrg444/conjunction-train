"""
FastAPI inference server for the Fusion-Emotion model.

POST /predict
  body: multipart/form-data
    video_file: video (mp4/avi) OPTIONAL
    audio_file: audio wav REQUIRED
Returns JSON with probabilities for 6 emotions.

Assumes the checkpoint weights are mounted at /opt/model/best.pt
and temp_corrected_train_fusion_model.py is present on PYTHONPATH.
"""

import io
import tempfile
import subprocess
from typing import Dict

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.requests import Request

# Bring training script directory into path so we can import FusionModel
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))  # So we can `import temp_corrected_train_fusion_model`

try:
    from temp_corrected_train_fusion_model import FusionModel, EMOTION_LABELS
except Exception as e:
    raise RuntimeError(f"Could not import FusionModel: {e}")

CHECKPOINT_PATH = Path("/opt/model/best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Fusion-Emotion Inference API", version="1.0.0")

# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------
def load_model() -> FusionModel:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    # Stubs for required args (not used during inference)
    model = FusionModel(
        num_classes=len(EMOTION_LABELS),
        video_checkpoint_path="",  # not used (we saved post-fusion checkpoint)
        hubert_checkpoint_path="",
        fusion_dim=512,
        dropout=0.5,
    )
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


model: FusionModel = load_model()

# -------------------------------------------------------------------------
# Health probe
# -------------------------------------------------------------------------
@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok"}

# -------------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    video_file: UploadFile = File(None),
) -> JSONResponse:
    try:
        # Save uploaded files to temp paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
            audio_bytes = await audio_file.read()
            audio_tmp.write(audio_bytes)
            audio_path = audio_tmp.name

        video_path = None
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_tmp:
                video_bytes = await video_file.read()
                video_tmp.write(video_bytes)
                video_path = video_tmp.name

        # -----------------------------------------------------------------
        # Feature extraction (re-use CLI helper script from repo)
        # For brevity we call an external script; adjust path as needed.
        # -----------------------------------------------------------------
        extract_script = ROOT_DIR / "scripts" / "extract_inference_features.py"
        if not extract_script.exists():
            raise RuntimeError("Feature extraction script not found.")

        # The script should output two tensors serialized with torch.save
        # for audio_features.pt and video_features.pt
        with tempfile.TemporaryDirectory() as feat_dir:
            cmd = [
                "python",
                str(extract_script),
                "--audio", audio_path,
                "--video", video_path or "",
                "--out_dir", feat_dir,
            ]
            subprocess.check_call(cmd)

            audio_tensor = torch.load(Path(feat_dir) / "audio.pt", map_location=device)
            video_tensor = torch.load(Path(feat_dir) / "video.pt", map_location=device)

        # Dummy attention mask (all ones)
        audio_att = torch.ones_like(audio_tensor, dtype=torch.long)

        # Batch dimension
        video_tensor = video_tensor.unsqueeze(0).to(device) if video_tensor is not None else None
        audio_tensor = audio_tensor.unsqueeze(0).to(device)
        audio_att = audio_att.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(video_tensor, audio_tensor, audio_att)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        result = {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
        return JSONResponse(result)

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
