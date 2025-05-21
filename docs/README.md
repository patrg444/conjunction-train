# Fusion-Emotion Multimodal Model – AWS Marketplace Deployment Guide

## Overview

The Fusion-Emotion model is a multimodal deep learning API for emotion recognition from video and audio. It leverages R3D-18 for video and HuBERT for audio, fusing both modalities for robust inference. This package is delivered as a Dockerized SageMaker endpoint, ready for deployment via AWS Marketplace.

---

## Architecture

- **Model:** R3D-18 (video) + HuBERT (audio) fusion
- **API:** FastAPI/Uvicorn, exposes `/ping` (health) and `/invocations` or `/predict` (inference)
- **Input:** Multipart form-data with video and/or audio files, or JSON with S3 URIs.
- **Output:** JSON with emotion predictions, confidence scores, and QA flags.

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
   # Replace <YOUR_SAGEMAKER_ROLE_ARN> with your actual SageMaker execution role ARN
   # The ECR_IMAGE_URI and S3_MODEL_DATA_URI are typically auto-filled if subscribing from Marketplace.
   # If deploying manually, use the URIs from the build_and_push.sh script output.
   # Example URIs from a sample build (replace with your latest):
   # ECR_IMAGE_URI="324037291814.dkr.ecr.us-west-2.amazonaws.com/fusion-emotion:latest-20250521014109"
   # S3_MODEL_DATA_URI="s3://fusion-emotion-model-artifacts-324037291814/model/model.tar.gz"

   aws sagemaker create-model \
     --model-name fusion-emotion \
     --primary-container Image="324037291814.dkr.ecr.us-west-2.amazonaws.com/fusion-emotion:latest-20250521014109",ModelDataUrl="s3://fusion-emotion-model-artifacts-324037291814/model/model.tar.gz" \
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
   *Wait until status is "InService".*

---

## Sample Inference Request

Replace `<ENDPOINT_NAME>` and `<REGION>` with your specific details.

```bash
# Example using multipart/form-data with local files:
curl -X POST "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/fusion-emotion-endpoint/invocations" \
  -H "Content-Type: multipart/form-data" \
  -F "video_file=@./ravdess_sample.mp4" \
  -F "audio_file=@./ravdess_sample.wav"

# Example for video-only (audio will be extracted):
# curl -X POST "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/fusion-emotion-endpoint/invocations" \
#  -H "Content-Type: multipart/form-data" \
#  -F "video_file=@./ravdess_sample.mp4"
```

**Sample Response:**
```json
{
    "angry": 0.0014676972059532999,
    "disgust": 0.0001139638916356489,
    "fearful": 0.0010991392191499472,
    "happy": 0.9880199432373047,
    "neutral": 0.008063508197665215,
    "sad": 0.0012357085943222046,
    "qa_flags": {
        "audio_clipping_ratio": 0.0,
        "audio_silence_ratio": 0.1,
        "video_face_detection_confidence": 0.99,
        "video_blur_level": 0.05,
        "modality_consistency_score": 0.85,
        "overall_confidence": 0.988
    },
    "dominant_emotion": "happy"
}
```
*(Note: Scores and QA flags are illustrative)*

---

## Input/Output Schema

- **Input:**
  - `multipart/form-data`:
    - `audio_file`: WAV file (Optional if `video_file` is provided and audio extraction is intended)
    - `video_file`: MP4, FLV, AVI file (Optional if `audio_file` is provided)
    *At least one of `audio_file` or `video_file` must be provided.*
  - `application/json`:
    ```json
    {
        "s3_audio_uri": "s3://bucket/path/to/audio.wav", // Optional
        "s3_video_uri": "s3://bucket/path/to/video.mp4"  // Optional
    }
    ```
    *At least one of `s3_audio_uri` or `s3_video_uri` must be provided. If only video URI is given, audio will be extracted.*

- **Output:** `application/json`
  - A flat JSON object where keys are emotion labels (`angry`, `disgust`, `fearful`, `happy`, `neutral`, `sad`) and values are their corresponding float scores (probabilities).
  - Includes a `qa_flags` object with various quality and diagnostic metrics.
  - Includes `dominant_emotion` string.

---

## Model Evaluation

**Validation Results (Fusion Model, Video+Audio, Macro Average):**
- **Best Validation Accuracy:** 89.59%
- **Macro F1-score:** 0.90
- **Macro Precision:** 0.90
- **Macro Recall:** 0.89
- **Validation set size:** 893 samples

**Per-class F1-scores:**
- Angry: 0.95
- Disgust: 0.94
- Fearful: 0.83
- Happy: 0.95
- Neutral: 0.89
- Sad: 0.81

*Model: fusion_model_20250427_053706_best.pt (trained with temp_corrected_train_fusion_model.py)*

---

## Performance

Benchmark tests conducted on `ml.g4dn.xlarge` instance in `us-west-2` with 50 requests at 20 concurrent workers.

- **RAVDESS Sample (Audio + Video):**
  - Average Latency: ~21.1 seconds
  - Throughput: ~0.80 requests/second
- **CREMA-D Sample (Audio + Video):**
  - Average Latency: ~8.2 seconds
  - Throughput: ~2.06 requests/second
- **RAVDESS Sample (Video-Only, audio extracted by server):**
  - Average Latency: ~21.0 seconds
  - Throughput: ~0.80 requests/second

*These metrics were last updated on 2025-05-21. To refresh, execute the benchmark using `python scripts/benchmark_sagemaker_endpoint.py ...` and update the values above from the generated `benchmark_results.json` file.*

---

## Key Use-Cases

-	Real-time customer-support sentiment
Pipe call-center audio or webcam feeds into the API to tag each interaction as Angry, Happy, Neutral, Sad, Fearful, or Surprised—triage frustrated callers and route them to senior agents automatically.
-	In-cabin driver-state monitoring
Automakers and fleet-safety vendors stream dash-cam video + cabin-mic audio to detect drowsiness-linked emotions (Sad / Neutral) or road-rage cues (Angry) and trigger in-car alerts.
-	Video-content emotion tagging
Media platforms batch-process YouTube/Twitch clips to index dominant on-screen emotion per timestamp—powering mood-based search (“show me the happiest moments”) and brand-safety filters.
-	Live virtual-event engagement analytics
Conference or webinar hosts track audience webcams (with consent) to display real-time “emotion heat-maps,” revealing when keynote slides evoke Surprise or when attention drops to Neutral.
-	Tele-health & mental-wellness monitoring
Telemedicine providers enrich session notes with emotion timelines, helping clinicians spot shifts toward Sad or Fearful affect that may warrant intervention.
-	Adaptive e-learning & coaching
EdTech platforms adjust lesson pacing or send supportive prompts when students’ webcams register extended Frustration/Anger or disengaged Neutral states.
-	Recruiting & HR interview insights
Overlay emotion analytics on recorded interviews to give hiring managers objective cues on candidate confidence (Happy / Surprised) or stress (Fearful / Sad).
-	Interactive gaming & XR immersion
Games and VR experiences query the API to adapt NPC behavior or scene difficulty in response to players’ real-time facial and vocal emotions, boosting immersion.

---

## Security & Compliance

- No persistent storage; all data is ephemeral
- No PII is stored or logged
- Container disables external network access (by default, SageMaker model containers do not have internet egress unless configured)
- SBOM and vulnerability scan report included

---

## Support

For support, contact: info@cliptvideos.com

---

## License

See LICENSE.txt for terms.
