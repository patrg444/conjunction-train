#!/usr/bin/env python3
"""
live_inference.py
-----------------
Real-time webcam + microphone emotion prediction using the
RAVDESS/CREMA-D FusionModel.

• Video captured via OpenCV
• Audio captured via sounddevice (16 kHz mono rolling buffer)
• Video features -> FaceNetExtractor
• Audio features -> HuBERT large
• Outputs top-1 emotion and probability overlayed on the webcam window.

Run
----
PYTHONPATH=. python live_inference.py --device cuda
Press “q” to quit.
"""

import cv2
import queue
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
import cv2
# (FaceNet extractor removed – we now feed raw clips to R3D-18)
# import FusionModel + labels
from temp_corrected_train_fusion_model import FusionModel, EMOTION_LABELS

# Load OpenCV's default Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

AUDIO_SR = 16000
AUDIO_BUFFER_SEC = 2.0   # rolling window
AUDIO_HOP_SEC = 0.5      # how often we run HuBERT on audio
VIDEO_INTERVAL = 3       # process every Nth frame

# ------------- video-clip parameters -------------
CLIP_LEN = 16            # number of consecutive frames per clip
FRAME_SIZE = 112         # spatial resolution expected by R3D-18
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- audio thread --------------------------- #
class AudioBuffer:
    def __init__(self, sr=AUDIO_SR, seconds=AUDIO_BUFFER_SEC):
        self.sr = sr
        self.max_len = int(sr * seconds)
        self.buffer = deque(maxlen=self.max_len)
        self.lock = threading.Lock()

    def audio_callback(self, indata, frames, time_info, status):
        with self.lock:
            self.buffer.extend(indata[:, 0].tolist())  # mono

    def get_waveform(self):
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return torch.tensor(np.array(self.buffer, dtype=np.float32))

# --------------------------- helpers --------------------------- #
def extract_audio_features(waveform, model, processor):
    inputs = processor(waveform.numpy(), sampling_rate=AUDIO_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device),
                        attention_mask=inputs.attention_mask.to(device) if "attention_mask" in inputs else None)
    return outputs.last_hidden_state[0].cpu()  # [T, 1024]


def main():
    print(f"Device: {device}")

    # models
    clip_buffer = deque(maxlen=CLIP_LEN)  # rolling frame buffer for R3D

    fusion_model = FusionModel(
        num_classes=len(EMOTION_LABELS),
        video_checkpoint_path="fusion_emotion_model_marketplace_package/model_weights/slowfast_emotion_20250425_201936_best.pt",
        hubert_checkpoint_path="fusion_emotion_model_marketplace_package/model_weights/hubert-ser-epoch=09-val_uar=0.75.ckpt",
        fusion_dim=512,
        dropout=0.5,
    )
    fusion_state = torch.load("fusion_emotion_model_marketplace_package/model_weights/fusion_model_20250427_053706_best.pt",
                              map_location=device)
    fusion_model.load_state_dict(fusion_state, strict=False)
    fusion_model.to(device).eval()

    # audio stream
    audio_buf = AudioBuffer()
    stream = sd.InputStream(channels=1, samplerate=AUDIO_SR, callback=audio_buf.audio_callback)
    stream.start()

    # video stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    last_audio_time = 0
    last_audio_wave = torch.zeros((1, 1))  # [1, T]
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed; exiting.")
            break

        # build/maintain 16-frame clip for R3D-18
        if frame_idx % VIDEO_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_res = cv2.resize(frame_rgb, (128, 128))
            h0 = (128 - FRAME_SIZE) // 2
            frame_crop = frame_res[h0 : h0 + FRAME_SIZE, h0 : h0 + FRAME_SIZE]
            tensor = (
                torch.tensor(frame_crop, dtype=torch.float32)
                .permute(2, 0, 1)
                .div(255.0)
            )
            tensor = (tensor - MEAN) / STD
            clip_buffer.append(tensor)

        if len(clip_buffer) == CLIP_LEN:
            clip = torch.stack(list(clip_buffer), dim=0).unsqueeze(0)  # [1,16,3,112,112]
        else:
            clip = torch.zeros(
                (1, CLIP_LEN, 3, FRAME_SIZE, FRAME_SIZE), dtype=torch.float32
            )

        # audio waveform every AUDIO_HOP_SEC
        if time.time() - last_audio_time >= AUDIO_HOP_SEC:
            wav = audio_buf.get_waveform()
            if wav is not None and len(wav) > AUDIO_SR * 0.2:  # at least 0.2 sec
                last_audio_wave = wav.unsqueeze(0)  # [1, T]
                last_audio_time = time.time()

        audio_input = last_audio_wave if last_audio_wave.numel() else torch.zeros((1, 1))
        audio_attention = torch.ones_like(audio_input, dtype=torch.long)

        with torch.no_grad():
            logits = fusion_model(
                clip.to(device),
                audio_input.to(device),
                audio_attention.to(device),
            )
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        label = EMOTION_LABELS[top_idx]
        prob = probs[top_idx]

        # overlay: show all class scores
        overlay_text = f"{label}: {prob:.2f}"
        for i, (lbl, p) in enumerate(zip(EMOTION_LABELS, probs)):
            overlay_text += f"\n{lbl}: {p:.2f}"
        y0 = 30
        for i, (lbl, p) in enumerate(zip(EMOTION_LABELS, probs)):
            text = f"{lbl}: {p:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y0 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if i == top_idx else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw a box around the detected face (using Haar cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow("Fusion Emotion Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
