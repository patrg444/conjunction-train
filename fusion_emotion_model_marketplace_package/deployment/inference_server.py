"""
FastAPI inference server for the Fusion-Emotion model.

POST /invocations or /predict
  Can accept either:
  1. multipart/form-data:
    audio_file: audio wav (REQUIRED if video_file is not for video-only mode)
    video_file: video (mp4/avi/flv) (REQUIRED if audio_file is not provided, or for video modality)
  2. application/json:
    body: {
        "s3_audio_uri": "s3://bucket/path/to/audio.wav", (REQUIRED if s3_video_uri is not for video-only)
        "s3_video_uri": "s3://bucket/path/to/video.mp4" (REQUIRED if s3_audio_uri is not provided, or for video modality)
    }
If only video_file or s3_video_uri is provided, audio will be extracted from it.

Returns JSON with probabilities for 6 emotions and QA flags.
"""

import io
import tempfile
import subprocess # Kept for potential ffmpeg direct call if ffmpeg-python fails
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from urllib.parse import urlparse
import random

import torch
import torch.nn.functional as F # For padding audio
import torchaudio
from torchvision import transforms
import cv2 # OpenCV for video processing
import numpy as np
import boto3
from PIL import Image # Import Pillow Image
from pydantic import BaseModel, validator
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse

# Attempt to import ffmpeg, will be used for audio extraction
try:
    import ffmpeg
except ImportError:
    print("WARNING: ffmpeg-python not installed. Audio extraction from video-only input will fail.")
    ffmpeg = None


# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR.parent))
sys.path.append(str(ROOT_DIR))

# --- Model and Feature Extractor Imports ---
try:
    from scripts.qa_flag_engine import QAFlagEngine
except ImportError:
    from qa_flag_engine import QAFlagEngine

try:
    from temp_corrected_train_fusion_model import FusionModel, EMOTION_LABELS
except ImportError:
    try:
        from temp_corrected_train_fusion_model import FusionModel, EMOTION_LABELS
    except ImportError as e:
        raise RuntimeError(f"Could not import FusionModel: {e}.")

from transformers import AutoFeatureExtractor, HubertModel

# --- Configuration & Globals ---
CHECKPOINT_PATH = Path(os.getenv("MODEL_PATH", "/opt/ml/model")) / "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S3_CLIENT = boto3.client("s3")

# Video processing parameters (from temp_corrected_train_fusion_model.py)
VIDEO_FRAMES = 48
VIDEO_IMG_SIZE = 112
VIDEO_MEAN = [0.485, 0.456, 0.406]
VIDEO_STD = [0.229, 0.224, 0.225]

# Audio processing parameters (from temp_corrected_train_fusion_model.py)
AUDIO_HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
AUDIO_TARGET_SR = 16000
AUDIO_MAX_LEN_SECONDS = 15.6 # Approx, from max_audio_len = 250000 in training
AUDIO_MAX_SAMPLES = int(AUDIO_TARGET_SR * AUDIO_MAX_LEN_SECONDS)


# --- QA Engine ---
qa_config_path = ROOT_DIR / "configs" / "qa_flags_config.yaml"
if not qa_config_path.exists():
    alt_qa_config_path = Path("configs") / "qa_flags_config.yaml"
    if alt_qa_config_path.exists(): qa_config_path = alt_qa_config_path
    else: raise FileNotFoundError(f"QA config not found at {qa_config_path} or {alt_qa_config_path}")
QA_FLAGGER = QAFlagEngine(str(qa_config_path))

# --- FastAPI App ---
app = FastAPI(title="Fusion-Emotion Inference API", version="1.1.0") # Version bump

# --- Pydantic model for S3 input ---
class S3InputPayload(BaseModel):
    s3_audio_uri: Optional[str] = None
    s3_video_uri: Optional[str] = None

    @validator('s3_audio_uri', 's3_video_uri', pre=True, always=True)
    def check_at_least_one_uri(cls, v, values):
        if not values.get('s3_audio_uri') and not values.get('s3_video_uri'):
            if 's3_audio_uri' in values and 's3_video_uri' in values: # only raise if both were attempted and failed
                 raise ValueError('At least one of s3_audio_uri or s3_video_uri must be provided.')
        return v
        
    @validator('s3_audio_uri', 's3_video_uri')
    def validate_s3_uri_format(cls, v):
        if v is not None and not v.startswith('s3://'):
            raise ValueError('S3 URI must start with s3://')
        return v

# --- Video Preprocessing ---
def get_video_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=VIDEO_MEAN, std=VIDEO_STD)
    ])

def load_video_frames_r3d18(video_path: str, num_frames: int, img_size: int) -> Optional[torch.Tensor]:
    if not video_path or not Path(video_path).exists():
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    transform = get_video_transforms(img_size)
    frames_list = []
    
    # Ensure we sample exactly num_frames, even if video is shorter
    # Linspace sampling, similar to training
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            if frames_list: # Repeat last frame if read fails mid-sequence
                # The line `frame_pil = frames_list[-1]._to_pil_image()` was from a misremembered version or different file.
                # The actual line from the file content that caused the error was:
                # frame_pil = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
                # This is correct usage assuming 'Image' is imported from PIL.
                frame_pil = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            else: # If first frame fails or no frames yet, use black frame
                frame_pil = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(cv2.resize(frame_rgb, (img_size, img_size)))
        
        frames_list.append(transform(frame_pil))
    
    cap.release()
    
    if not frames_list: return None
    return torch.stack(frames_list) # Shape: [T, C, H, W]

# --- Audio Preprocessing ---
AUDIO_PROCESSOR = AutoFeatureExtractor.from_pretrained(AUDIO_HUBERT_MODEL_NAME)
AUDIO_MODEL = HubertModel.from_pretrained(AUDIO_HUBERT_MODEL_NAME).to(DEVICE).eval()

def extract_audio_features_hubert(audio_path: str) -> Optional[torch.Tensor]:
    if not audio_path or not Path(audio_path).exists():
        return None
    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != AUDIO_TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_TARGET_SR)
            waveform = resampler(waveform)
        
        waveform_np = waveform.squeeze(0).cpu().numpy()
        
        # Process with feature extractor (handles padding/truncation)
        processed_audio = AUDIO_PROCESSOR(
            waveform_np,
            sampling_rate=AUDIO_TARGET_SR,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=AUDIO_MAX_SAMPLES 
        )
        input_values = processed_audio.input_values.to(DEVICE)
        attention_mask = processed_audio.attention_mask.to(DEVICE) if 'attention_mask' in processed_audio else None

        with torch.no_grad():
            outputs = AUDIO_MODEL(input_values, attention_mask=attention_mask)
        
        # Pool hidden states (mean pooling is common for HuBERT embeddings for SER)
        # Taking the last hidden state and mean pooling across the time dimension
        # Ensure attention_mask is used for pooling if applicable
        if attention_mask is not None:
            sum_hidden_states = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
            sum_attention_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            pooled_features = sum_hidden_states / sum_attention_mask
        else:
            pooled_features = torch.mean(outputs.last_hidden_state, dim=1)
        
        return pooled_features.squeeze(0).cpu() # Return [hidden_dim]

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

# --- Model Loading ---
FUSION_MODEL: FusionModel = None
def load_fusion_model():
    global FUSION_MODEL
    # Resolve checkpoint path with env var override, standard and glob fallback
    model_filename = os.getenv("MODEL_FILENAME")
    model_dir = CHECKPOINT_PATH.parent
    if model_filename:
        env_path = model_dir / model_filename
        if env_path.exists():
            current_checkpoint_path = env_path
        else:
            print(f"WARNING: MODEL_FILENAME env var provided but {env_path} not found. Continuing fallback.")
            model_filename = None
    if not model_filename:
        if CHECKPOINT_PATH.exists():
            current_checkpoint_path = CHECKPOINT_PATH
        else:
            alt_checkpoint_path = Path("/opt/model/best.pt") # Common alternative for local/older setups
            if alt_checkpoint_path.exists():
                current_checkpoint_path = alt_checkpoint_path
            else:
                found = list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.ckpt"))
                if found:
                    current_checkpoint_path = found[0]
                    print(f"WARNING: Using fallback checkpoint {current_checkpoint_path}")
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}, {alt_checkpoint_path}, or in {model_dir}")

    # Initialize with dummy paths as they are not used when loading a full fused model state_dict
    # The original FusionModel might sys.exit if these paths are not valid and it tries to load them.
    # We use strict=False for loading the final fused model.
    FUSION_MODEL = FusionModel(
        num_classes=len(EMOTION_LABELS),
        video_checkpoint_path="dummy_video_path.pt", # Not actually loaded
        hubert_checkpoint_path="dummy_audio_path.ckpt", # Not actually loaded
        fusion_dim=512, dropout=0.5
    )
    print(f"Loading model checkpoint from: {current_checkpoint_path}")
    state = torch.load(current_checkpoint_path, map_location=DEVICE)
    if 'state_dict' in state: state = state['state_dict']
    state = {k.replace("model.", ""): v for k, v in state.items()} # For PL checkpoints
    
    FUSION_MODEL.load_state_dict(state, strict=False) # strict=False for fused model
    FUSION_MODEL.eval().to(DEVICE)
    print("Fusion model loaded successfully.")

load_fusion_model() # Load on startup

# --- Helper to parse S3 URI ---
def parse_s3_uri_helper(s3_uri: str) -> Tuple[str, str]: # Renamed to avoid conflict
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3": raise ValueError(f"Invalid S3 URI: {s3_uri}.")
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    if not key: raise ValueError(f"S3 URI missing key: {s3_uri}.")
    return bucket, key

# --- Audio Extraction from Video ---
def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    if not ffmpeg:
        print("ffmpeg-python not available, cannot extract audio.")
        return False
    try:
        print(f"Extracting audio from {video_path} to {output_audio_path}...")
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ar=str(AUDIO_TARGET_SR), ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        print("Audio extraction successful.")
        return True
    except Exception as e:
        print(f"Error extracting audio using ffmpeg-python: {e}")
        # Fallback to direct subprocess call if needed, though ffmpeg-python is preferred
        try:
            print("Attempting ffmpeg CLI fallback for audio extraction...")
            command = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", str(AUDIO_TARGET_SR), "-ac", "1",
                output_audio_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Audio extraction with ffmpeg CLI successful.")
            return True
        except Exception as e_sub:
            print(f"ffmpeg CLI fallback also failed: {e_sub}")
            return False

# --- Prediction Endpoint ---
@app.post("/invocations", summary="Predict emotions from audio/video")
@app.post("/predict", summary="Predict emotions (alias)")
async def predict_endpoint_handler( # Renamed to avoid conflict
    request: Request,
    s3_payload: Optional[S3InputPayload] = Body(None),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None)
) -> JSONResponse:
    print("PREDICT_ENDPOINT_HANDLER CALLED")
    temp_files_to_clean: List[str] = []
    audio_path_local: Optional[str] = None
    video_path_local: Optional[str] = None

    try:
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            if not s3_payload or (not s3_payload.s3_audio_uri and not s3_payload.s3_video_uri):
                raise HTTPException(status_code=400, detail="Missing s3_audio_uri or s3_video_uri in JSON.")
            
            if s3_payload.s3_audio_uri:
                bucket, key = parse_s3_uri_helper(s3_payload.s3_audio_uri)
                suffix = Path(key).suffix or ".wav"
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                S3_CLIENT.download_file(bucket, key, temp_audio_file.name)
                audio_path_local = temp_audio_file.name
                temp_files_to_clean.append(audio_path_local)
                temp_audio_file.close()
            
            if s3_payload.s3_video_uri:
                bucket, key = parse_s3_uri_helper(s3_payload.s3_video_uri)
                suffix = Path(key).suffix or ".mp4"
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                S3_CLIENT.download_file(bucket, key, temp_video_file.name)
                video_path_local = temp_video_file.name
                temp_files_to_clean.append(video_path_local)
                temp_video_file.close()

        elif "multipart/form-data" in content_type:
            if not audio_file and not video_file:
                raise HTTPException(status_code=400, detail="Missing audio_file or video_file in multipart.")

            if audio_file:
                suffix = Path(audio_file.filename).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await audio_file.read())
                    audio_path_local = tmp.name
                temp_files_to_clean.append(audio_path_local)
            
            if video_file:
                suffix = Path(video_file.filename).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await video_file.read())
                    video_path_local = tmp.name
                temp_files_to_clean.append(video_path_local)
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {content_type}.")

        # Video-only mode: extract audio if not provided
        if video_path_local and not audio_path_local:
            print(f"Video-only mode: Attempting to extract audio from {video_path_local}")
            extracted_audio_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            if extract_audio_from_video(video_path_local, extracted_audio_temp_file.name):
                audio_path_local = extracted_audio_temp_file.name
                temp_files_to_clean.append(audio_path_local)
            else:
                print(f"Failed to extract audio from {video_path_local}. Proceeding without audio.")
            extracted_audio_temp_file.close()


        if not audio_path_local and not video_path_local: # Should not happen if validators work
             raise HTTPException(status_code=400, detail="No audio or video input provided.")

        # --- Feature Extraction ---
        video_tensor: Optional[torch.Tensor] = None
        if video_path_local:
            video_tensor = load_video_frames_r3d18(video_path_local, VIDEO_FRAMES, VIDEO_IMG_SIZE)
            if video_tensor is not None:
                 video_tensor = video_tensor.unsqueeze(0).to(DEVICE) # Add batch dim
            else:
                print(f"Warning: Could not process video from {video_path_local}")
        
        # For FusionModel, even if one modality fails, we might want to proceed with the other.
        # The model's forward pass should handle None for a missing modality.
        # However, the training script's FusionModel expects both inputs.
        # For inference, we need to ensure the model's forward method can handle a None video_input.
        # The current FusionModel forward is: model(video_data, audio_input_values, audio_attention_mask)
        # It will fail if video_data is None and it tries to pass it to self.video_embedder
        # Let's provide zero tensor if video processing fails or no video.
        if video_tensor is None:
            print("No video features to use. Using zero tensor for video input.")
            # Shape: [Batch, Time, Channels, Height, Width] for R3D-18
            video_tensor = torch.zeros(1, VIDEO_FRAMES, 3, VIDEO_IMG_SIZE, VIDEO_IMG_SIZE, device=DEVICE)


        audio_features: Optional[torch.Tensor] = None
        if audio_path_local:
            # The training script's FusionDataset uses AutoFeatureExtractor which returns input_values and attention_mask
            # The HubertSER model within FusionModel then takes these.
            # For direct inference, we need to replicate this.
            waveform, sr = torchaudio.load(audio_path_local)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != AUDIO_TARGET_SR:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=AUDIO_TARGET_SR)(waveform)
            
            processed_audio = AUDIO_PROCESSOR(
                waveform.squeeze(0).cpu().numpy(), sampling_rate=AUDIO_TARGET_SR, return_tensors="pt",
                padding='max_length', truncation=True, max_length=AUDIO_MAX_SAMPLES
            )
            audio_input_values = processed_audio.input_values.to(DEVICE)
            audio_attention_mask = processed_audio.attention_mask.to(DEVICE) if 'attention_mask' in processed_audio else torch.ones_like(audio_input_values, device=DEVICE)
        else:
            # If no audio, create dummy tensors for the model
            print("No audio features to use. Using zero tensor for audio input.")
            audio_input_values = torch.zeros(1, AUDIO_MAX_SAMPLES, device=DEVICE) # Batch, MaxSamples
            audio_attention_mask = torch.zeros(1, AUDIO_MAX_SAMPLES, device=DEVICE, dtype=torch.long)


        # --- Prediction ---
        with torch.no_grad():
            # Ensure video_tensor is not None for the model
            # The FusionModel's forward pass: forward(self, video_input, audio_input_values, audio_attention_mask)
            logits = FUSION_MODEL(video_tensor, audio_input_values, audio_attention_mask)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        result = {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}

        turn_json = {"timestamp": "0", "speaker": "agent", "faceEmotion": result.copy(), "voiceTone": {}, "fusionScore": max(probs) if probs else 0.0}
        enriched = QA_FLAGGER.process_turn(turn_json)
        result.update(enriched)
        return JSONResponse(result)

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {str(fnfe)}")
        raise HTTPException(status_code=500, detail=f"A required file was not found: {str(fnfe)}")
    except Exception as ex:
        print(f"Unhandled Exception: {str(ex)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(ex)}")
    finally:
        for temp_file_path in temp_files_to_clean:
            try:
                if Path(temp_file_path).exists(): os.remove(temp_file_path)
            except Exception as e_clean:
                print(f"Error cleaning temp file {temp_file_path}: {e_clean}")

# --- Health Probes ---
@app.get("/ping", summary="Health check")
async def ping(): return {"status": "ok", "message": "Server is running"}

@app.get("/health", summary="Detailed health check")
async def health(): return {"status": "healthy", "model_ready": FUSION_MODEL is not None and CHECKPOINT_PATH.exists()}

@app.get("/version", summary="Get API version")
async def version_endpoint(): return {"version": app.version}
