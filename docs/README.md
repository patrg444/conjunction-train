# Fusion-Emotion Multimodal Model – AWS Marketplace Deployment Guide

## Overview

The Fusion-Emotion model is a multimodal deep learning API for emotion recognition from video and audio. It leverages R3D-18 for video and HuBERT for audio, fusing both modalities for robust inference. This package is delivered as a Dockerized SageMaker endpoint, ready for deployment via AWS Marketplace.

---

## Architecture

- **Model:** R3D-18 (video) + HuBERT (audio) fusion
- **API:** FastAPI/Uvicorn, exposes `/ping` (health) and `/invocations` (inference)
- **Input:** Multipart form-data with video and audio files
- **Output:** JSON with emotion predictions and confidence scores

---

## AWS Costs

| Instance Type      | vCPU | GPU      | Memory (GiB) | On-Demand Price (USD/hr) |
|--------------------|------|----------|--------------|--------------------------|
| ml.g4dn.xlarge     | 4    | 1x T4    | 16           | ~$0.68                   |
| ml.g5.xlarge       | 4    | 1x A10G  | 16           | ~$1.20                   |

*See [AWS Pricing](https://aws.amazon.com/sagemaker/pricing/) for latest rates.*

---

## QuickStart: Deploy with AWS CLI

1. **Create SageMaker Model from Marketplace Package**

   ```bash
   aws sagemaker create-model \
     --model-name fusion-emotion \
     --primary-container Image=<ECR_IMAGE_URI>,ModelDataUrl=<S3_MODEL_DATA_URI> \
     --execution-role-arn <YOUR_SAGEMAKER_ROLE_ARN> \
     --region us-west-2
   ```

2. **Create Endpoint Config**

   ```bash
   aws sagemaker create-endpoint-config \
     --endpoint-config-name fusion-emotion-config \
     --production-variants VariantName=AllTraffic,ModelName=fusion-emotion,InitialInstanceCount=1,InstanceType=ml.g4dn.xlarge \
     --region us-west-2
   ```

3. **Create Endpoint**

   ```bash
   aws sagemaker create-endpoint \
     --endpoint-name fusion-emotion-endpoint \
     --endpoint-config-name fusion-emotion-config \
     --region us-west-2
   ```

4. **Check Endpoint Status**

   ```bash
   aws sagemaker describe-endpoint --endpoint-name fusion-emotion-endpoint --region us-west-2 --query EndpointStatus
   ```

---

## Sample Inference Request

```bash
curl -X POST "https://<ENDPOINT_URL>/invocations" \
  -F "video=@/path/to/video.mp4" \
  -F "audio=@/path/to/audio.wav"
```

**Response:**
```json
{
  "emotions": [
    {"label": "happy", "score": 0.92},
    {"label": "neutral", "score": 0.07},
    {"label": "sad", "score": 0.01}
  ]
}
```

---

## Input/Output Schema

- **Input:** `multipart/form-data`
  - `video`: MP4 file (required)
  - `audio`: WAV file (required)
- **Output:** `application/json`
  - `emotions`: List of objects with `label` (str) and `score` (float)

---

## Performance

- **Latency:** ~1.5s per request (g4dn.xlarge, 10s video/audio)
- **Throughput:** ~30 requests/minute (single instance)

---

## Security & Compliance

- No persistent storage; all data is ephemeral
- No PII is stored or logged
- Container disables external network access
- SBOM and vulnerability scan report included

---

## Support

For support, contact: [your-email@example.com]

---

## License

See LICENSE.txt for terms.
