# ─────────────────────────────────────────────────────────────
# CUDA-enabled PyTorch inference image for Fusion-Emotion model
# ─────────────────────────────────────────────────────────────
# NOTE: SageMaker GPU instances (ml.g4dn.*) currently support CUDA 11.8.
# We use the official PyTorch 2.1.0-cu118 runtime image.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ---------- Environment settings ----------
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System packages ----------
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /app

# ---------- Python deps ----------
COPY fusion_emotion_model_marketplace_package/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---------- Copy code ----------
# Copy only what we need to speed up build; adjust if other modules are required.
COPY fusion_emotion_model_marketplace_package/src/temp_corrected_train_fusion_model.py /app/
COPY fusion_emotion_model_marketplace_package/deployment/ /app/deployment/
COPY deployment_scripts/serve_entrypoint.sh /app/deployment/serve_entrypoint.sh
COPY fusion_emotion_model_marketplace_package/src/scripts/ /app/scripts/
COPY fusion_emotion_model_marketplace_package/configs/ /app/configs/
COPY ser_hubert/ /app/ser_hubert/
# Model weights will be injected at /opt/model during SageMaker container start.

# ---------- Expose & entrypoint ----------
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8080

ENTRYPOINT ["/app/deployment/serve_entrypoint.sh"]
CMD ["serve"]
