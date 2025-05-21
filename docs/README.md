# Fusion-Emotion Multimodal Model – AWS Marketplace Deployment Guide

## Overview

The Fusion-Emotion model is a versatile multimodal deep learning API for emotion recognition from video and audio. It leverages R3D-18 for video and HuBERT for audio, fusing both modalities for robust inference. This package is delivered as a Dockerized SageMaker endpoint, supporting both **high-throughput batch processing** of pre-recorded media and enabling **real-time emotion analysis** through client-side segmentation of live streams.

This guide provides instructions for deploying the SageMaker endpoint and integrating it into your workflows.

---

## Architecture

- **Model:** R3D-18 (video) + HuBERT (audio) fusion architecture.
- **API:** FastAPI/Uvicorn server running on SageMaker.
  - `/ping`: Health check.
  - `/invocations` (or `/predict` alias): Main inference endpoint.
- **Processing Modes Supported by the Endpoint:**
  - **Discrete File/Segment Processing**: Accepts individual audio/video files (or segments) per invocation, either via multipart/form-data or by referencing S3 URIs. This is the primary mode of the endpoint.
- **Input Types:**
  - `multipart/form-data`: For uploading local audio/video files directly.
  - `application/json`: For specifying S3 URIs of audio/video files.
- **Output:** JSON response containing:
  - Emotion probabilities (for 6 core emotions: angry, disgust, fearful, happy, neutral, sad).
  - A `dominant_emotion` field.
  - A `qa_flags` object with various quality and diagnostic metrics (e.g., audio silence ratio, face detection confidence).

---

## Use Cases & Processing Modes

This model can be adapted for a wide range of applications:

**1. Batch Processing of Pre-recorded Media:**
   Ideal for analyzing existing archives of video or audio content.
   - **How**: Use the SageMaker endpoint with SageMaker Batch Transform, or write scripts to iterate through your media files and call the endpoint for each.
   - **Example Use Cases from README**:
     - Video-content emotion tagging
     - Recruiting & HR interview insights (analysis of recorded interviews)
     - Enriching tele-health session notes from recordings

**2. Real-Time / Live Emotion Analysis:**
   Suitable for applications requiring immediate emotional feedback from live audio/video feeds.
   - **How**: Implement client-side logic to capture live audio/video, segment it into short, manageable chunks (e.g., 2-10 seconds), and send each chunk to the SageMaker endpoint for processing. The endpoint processes each segment and returns emotion scores for that chunk.
   - **Example Use Cases from README**:
     - Real-time customer-support sentiment
     - In-cabin driver-state monitoring
     - Live virtual-event engagement analytics
     - Adaptive e-learning & coaching
     - Interactive gaming & XR immersion
   - *(See "Achieving Real-Time Performance with Client-Side Segmentation" section below for more details and a conceptual client design.)*

---

## AWS Costs

| Instance Type      | vCPU | GPU      | Memory (GiB) | On-Demand Price (USD/hr) | Notes                                   |
|--------------------|------|----------|--------------|--------------------------|-----------------------------------------|
| ml.g4dn.xlarge     | 4    | 1x T4    | 16           | ~$0.68                   | Good for moderate batch or dev/test.    |
| ml.g5.xlarge       | 4    | 1x A10G  | 16           | ~$1.20                   | Recommended for lower latency real-time.|
| *Other g4dn/g5/p3/p4 instances can also be used depending on budget and throughput needs.* |

*See [AWS Pricing](https://aws.amazon.com/sagemaker/pricing/) for latest rates. Costs depend on instance type, endpoint uptime, and data transfer.*

---

## QuickStart: Deploy with AWS CLI

1.  **Subscribe to the Model Package in AWS Marketplace.** (This step is usually done via the AWS Console).
2.  **Create SageMaker Model from the Marketplace Package ARN**
    (The ModelPackageArn is provided after subscribing in Marketplace)
    ```bash
    aws sagemaker create-model \
      --model-name fusion-emotion-model \
      --primary-container ModelPackageName=<YOUR_MARKETPLACE_MODEL_PACKAGE_ARN> \
      --execution-role-arn <YOUR_SAGEMAKER_ROLE_ARN> \
      --region us-west-2 
    ```
    *Alternatively, if deploying manually using a pre-built image (not via Marketplace subscription):*
    ```bash
    # ECR_IMAGE_URI="324037291814.dkr.ecr.us-west-2.amazonaws.com/fusion-emotion:latest-20250521014109" (example)
    # S3_MODEL_DATA_URI="s3://fusion-emotion-model-artifacts-324037291814/model/model.tar.gz" (example)
    aws sagemaker create-model \
      --model-name fusion-emotion-model \
      --primary-container Image="<ECR_IMAGE_URI_FROM_YOUR_BUILD>",ModelDataUrl="<S3_MODEL_DATA_URI_FROM_YOUR_BUILD>" \
      --execution-role-arn <YOUR_SAGEMAKER_ROLE_ARN> \
      --region us-west-2
    ```

3.  **Create Endpoint Config**
    (Choose an appropriate `InstanceType` based on your needs, e.g., `ml.g4dn.xlarge` or `ml.g5.xlarge`)
    ```bash
    aws sagemaker create-endpoint-config \
      --endpoint-config-name fusion-emotion-config \
      --production-variants VariantName=AllTraffic,ModelName=fusion-emotion-model,InitialInstanceCount=1,InstanceType=ml.g4dn.xlarge \
      --region us-west-2
    ```

4.  **Create Endpoint**
    ```bash
    aws sagemaker create-endpoint \
      --endpoint-name fusion-emotion-endpoint \
      --endpoint-config-name fusion-emotion-config \
      --region us-west-2
    ```

5.  **Check Endpoint Status**
    ```bash
    aws sagemaker describe-endpoint --endpoint-name fusion-emotion-endpoint --region us-west-2 --query EndpointStatus
    ```
    *Wait until status is "InService". This can take 5-15 minutes.*

---

## Sample Inference Request

The endpoint name used below is `fusion-emotion-endpoint`. Replace with your endpoint name if different. The region is `us-west-2`.

**Using `multipart/form-data` (local files):**
```bash
curl -X POST "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/fusion-emotion-endpoint/invocations" \
  -H "Content-Type: multipart/form-data" \
  -F "video_file=@./ravdess_sample.mp4" \
  -F "audio_file=@./ravdess_sample.wav"
```

**Using `application/json` (S3 URIs):**
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name fusion-emotion-endpoint \
  --region us-west-2 \
  --content-type "application/json" \
  --body '{"s3_video_uri": "s3://your-bucket/path/to/video.mp4", "s3_audio_uri": "s3://your-bucket/path/to/audio.wav"}' \
  output.json
# View response: cat output.json
```

**Sample Response (illustrative):**
```json
{
    "angry": 0.0015,
    "disgust": 0.0001,
    "fearful": 0.0011,
    "happy": 0.9880,
    "neutral": 0.0081,
    "sad": 0.0012,
    "qa_flags": {
        "audio_clipping_ratio": 0.0,
        "audio_silence_ratio": 0.1,
        "video_face_detection_confidence": 0.99,
        "video_blur_level": 0.05,
        "modality_consistency_score": 0.85, // Example metric
        "overall_confidence": 0.988 // Example metric
    },
    "dominant_emotion": "happy"
}
```
*(Actual scores, QA flags, and their specific names/values will depend on the model version and QA engine configuration.)*

---

## Input/Output Schema

-   **Input:** The endpoint accepts requests via the `/invocations` or `/predict` paths.
    -   **Mode 1: `multipart/form-data`**
        -   `audio_file`: (Optional) A WAV audio file.
        -   `video_file`: (Optional) An MP4, FLV, or AVI video file.
        *At least one of `audio_file` or `video_file` must be provided. If only `video_file` is provided, audio will be extracted from it for multimodal analysis.*
    -   **Mode 2: `application/json`**
        A JSON body with the following structure:
        ```json
        {
            "s3_audio_uri": "s3://bucket/path/to/your-audio.wav", // Optional
            "s3_video_uri": "s3://bucket/path/to/your-video.mp4"  // Optional
        }
        ```
        *At least one of `s3_audio_uri` or `s3_video_uri` must be provided. If only `s3_video_uri` is given, audio will be extracted.*

-   **Output:** `application/json`
    A JSON object containing:
    -   Emotion probabilities: Keys are emotion labels (`angry`, `disgust`, `fearful`, `happy`, `neutral`, `sad`), values are float scores (0.0 to 1.0).
    -   `dominant_emotion`: String indicating the emotion with the highest score.
    -   `qa_flags`: An object containing various quality assessment metrics. The specific flags and their meanings are defined by the `qa_flags_config.yaml` used by the QAEngine. Example flags include `audio_clipping_ratio`, `video_face_detection_confidence`, etc.

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

Benchmark tests conducted on `ml.g4dn.xlarge` instance in `us-west-2` with 50 requests at 20 concurrent workers, using distinct audio/video samples. The endpoint processes the full duration of provided segments (up to internal limits like `AUDIO_MAX_SAMPLES`).

-   **RAVDESS Sample (Audio + Video, ~5s):**
    -   Average Latency: ~21.1 seconds
    -   Throughput: ~0.80 requests/second
-   **CREMA-D Sample (Audio + Video, ~3s):**
    -   Average Latency: ~8.2 seconds
    -   Throughput: ~2.06 requests/second
-   **RAVDESS Sample (Video-Only, audio extracted by server, ~5s):**
    -   Average Latency: ~21.0 seconds
    -   Throughput: ~0.80 requests/second

*These metrics were last updated on 2025-05-21. Latency includes data transfer, pre-processing, model inference, and post-processing within the endpoint. For real-time applications, client-side segmentation of live streams into short chunks (e.g., 2-10 seconds) is recommended. Per-segment latency will vary based on segment duration and instance type.*

---

## Achieving Real-Time Performance with Client-Side Segmentation

For live streaming use cases (e.g., webcam/microphone feeds), the client application should manage the stream segmentation and send these segments to the SageMaker endpoint. Here's a conceptual approach:

1.  **Capture & Buffer**:
    *   Use client-side libraries (e.g., OpenCV for video, PyAudio/Sounddevice for audio) to capture media.
    *   Maintain rolling buffers (e.g., `collections.deque`) for the last L seconds of audio and video data (e.g., L = 2-5 seconds).
2.  **Window Assembly & Synchronization**:
    *   Periodically (e.g., every S seconds, where S < L, creating overlap), extract the latest L-second window from both buffers.
    *   Ensure audio and video segments are reasonably synchronized (e.g., using timestamps).
3.  **Prepare Segment for Endpoint**:
    *   Save the L-second audio window as a temporary WAV file.
    *   Save the L-second video window as a temporary MP4 file (or other supported format).
4.  **Invoke Endpoint**:
    *   Send these temporary files to the SageMaker endpoint using `multipart/form-data`.
5.  **Process Response**:
    *   Receive emotion scores for that L-second window.
    *   Optionally apply smoothing (e.g., Exponential Moving Average) over a series of window predictions.
6.  **Repeat**: Continue capturing and sending new windows.

**Latency Considerations for Real-Time:**
*   The end-to-end user-visible latency will be approximately: `Segment Duration (L)` + `Hop/Stride Time (S)` (if waiting for next segment) + `Network Latency` + `Endpoint Processing Latency` (as benchmarked for short segments).
*   Choosing an appropriate instance type (e.g., `ml.g5.xlarge` for lower GPU latency) and optimizing segment duration are key.
*   The `live_inference.py` script in this repository (`fusion_emotion_model_marketplace_package/src/scripts/live_inference.py`) demonstrates local real-time inference with similar buffering logic; it can be adapted to call the SageMaker endpoint instead of a local model.

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

- No persistent storage within the endpoint; all data is ephemeral during processing.
- No PII is stored or logged by the inference server application. Standard SageMaker logs (invocation metadata, container logs if enabled) are managed by AWS.
- Container disables external network access by default when deployed via SageMaker.
- SBOM and vulnerability scan report can be provided upon request.

---

## Support

For support, contact: info@cliptvideos.com

---

## License

See LICENSE.txt for terms.
