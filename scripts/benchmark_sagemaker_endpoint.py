#!/usr/bin/env python3
import argparse
import boto3
import concurrent.futures
import datetime
import json
import logging
import os
import pathlib
import time
import uuid
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder # For multipart/form-data
from typing import Optional

# --- Configuration ---
DEFAULT_LOOPS = 50
DEFAULT_MAX_WORKERS = 20
DEFAULT_REGION = "us-west-2" # Fallback if not in env
LOCAL_CACHE_DIR = pathlib.Path.home() / ".codex_bench"
DEFAULT_OUTPUT_JSON = "benchmark_results.json"

# Public S3 URIs for default sample files
DEFAULT_SAMPLE_FILES_S3 = {
    "ravdess_video": "s3://codex-public-samples/Actor_01_01.mp4",
    "ravdess_audio": "s3://codex-public-samples/Actor_01_01.wav",
    "crema_d_video": "s3://codex-public-samples/1001_DFA_ANG_XX.flv",
    "crema_d_audio": "s3://codex-public-samples/1001_DFA_ANG_XX.wav",
}

# --- Logging Setup ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Suppress overly verbose boto3 logging
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- S3 Utilities ---
def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    if not key:
        raise ValueError(f"S3 URI missing key: {s3_uri}")
    return bucket, key

def download_s3_file(s3_client, s3_uri: str, local_path: pathlib.Path) -> None:
    try:
        bucket, key = parse_s3_uri(s3_uri)
        logging.info(f"Downloading {s3_uri} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logging.info(f"Successfully downloaded {local_path}")
    except Exception as e:
        logging.error(f"Failed to download {s3_uri}: {e}")
        raise

# --- Sample File Management ---
def prepare_sample_files(s3_client, local_cache_dir: pathlib.Path) -> dict:
    local_sample_paths = {}
    for name, s3_uri in DEFAULT_SAMPLE_FILES_S3.items():
        local_file_name = pathlib.Path(s3_uri).name
        local_path = local_cache_dir / local_file_name
        if not local_path.exists():
            logging.info(f"Sample file {local_path} not found in cache.")
            download_s3_file(s3_client, s3_uri, local_path)
        else:
            logging.info(f"Using cached sample file: {local_path}")
        local_sample_paths[name] = local_path
    return local_sample_paths

# --- SageMaker Invocation ---
def invoke_sagemaker_multipart(
    sagemaker_runtime_client, endpoint_name: str, audio_path: Optional[str], video_path: Optional[str]
) -> tuple[float, Optional[dict], Optional[str]]:
    fields = {}
    if not audio_path and not video_path:
        return 0, None, "At least one of audio_path or video_path must be provided."
    if audio_path:
        if not os.path.exists(audio_path):
            return 0, None, f"Audio file not found: {audio_path}"
        fields['audio_file'] = (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/wav')
    if video_path:
        if not os.path.exists(video_path):
            return 0, None, f"Video file not found: {video_path}"
        video_filename = os.path.basename(video_path)
        content_type = 'video/mp4' if video_filename.endswith('.mp4') else \
                       'video/x-flv' if video_filename.endswith('.flv') else \
                       'video/x-msvideo' if video_filename.endswith('.avi') else \
                       'application/octet-stream'
        fields['video_file'] = (video_filename, open(video_path, 'rb'), content_type)

    encoder = MultipartEncoder(fields=fields)
    
    start_time = time.perf_counter()
    try:
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=encoder.content_type,
            Body=encoder.to_string()
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        return latency_ms, response_body, None
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logging.error(f"Multipart request failed: {e}")
        return latency_ms, None, str(e)
    finally:
        # Ensure files are closed if opened directly by MultipartEncoder
        if 'audio_file' in fields and hasattr(fields['audio_file'][1], 'close'):
            fields['audio_file'][1].close()
        if 'video_file' in fields and hasattr(fields['video_file'][1], 'close'):
            fields['video_file'][1].close()


def invoke_sagemaker_s3(
    sagemaker_runtime_client, endpoint_name: str, s3_audio_uri: str, s3_video_uri: Optional[str]
) -> tuple[float, Optional[dict], Optional[str]]:
    payload = {"s3_audio_uri": s3_audio_uri}
    if s3_video_uri:
        payload["s3_video_uri"] = s3_video_uri
    
    body = json.dumps(payload).encode('utf-8')
    
    start_time = time.perf_counter()
    try:
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=body
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        response_body = json.loads(response['Body'].read().decode('utf-8'))
        return latency_ms, response_body, None
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logging.error(f"S3 URI request failed: {e}")
        return latency_ms, None, str(e)

# --- Sanity Check ---
def is_response_valid(response_body: Optional[dict]) -> bool:
    if response_body is None:
        return False
    # Example check: ensure top-level emotion keys are present and have float scores
    # This should align with the actual server output structure.
    # Based on inference_server.py, EMOTION_LABELS are top-level keys.
    emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad'] # From training script
    for label in emotion_labels:
        if label not in response_body or not isinstance(response_body[label], float):
            logging.warning(f"Invalid response structure: Missing or invalid type for emotion '{label}'. Response: {response_body}")
            return False
    return True


# --- Main Benchmark Logic ---
def run_benchmark_request(
    sagemaker_runtime_client, endpoint_name: str, region_name: str, # region_name not used by boto3 client if already configured
    mode: str, sample_pair_details: dict
) -> tuple[float, bool, Optional[str]]:
    
    latency_ms = 0
    error_message = None
    response_body = None

    if mode == "multipart":
        latency_ms, response_body, error_message = invoke_sagemaker_multipart(
            sagemaker_runtime_client,
            endpoint_name,
            sample_pair_details["audio_path"],
            sample_pair_details.get("video_path")
        )
    elif mode == "s3":
        latency_ms, response_body, error_message = invoke_sagemaker_s3(
            sagemaker_runtime_client,
            endpoint_name,
            sample_pair_details["s3_audio_uri"],
            sample_pair_details.get("s3_video_uri")
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if error_message:
        return latency_ms, False, error_message
    
    if not is_response_valid(response_body):
        logging.warning(f"Invalid response received: {response_body}")
        # Decide if this should count as a hard failure or just a warning
        return latency_ms, False, f"Invalid response structure: {json.dumps(response_body)}"

    return latency_ms, True, None


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Benchmark SageMaker Endpoint for Fusion-Emotion Model")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name (or set SAGEMAKER_ENDPOINT_NAME env var)")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION), help=f"AWS region (default: {DEFAULT_REGION} or AWS_REGION env var)")
    parser.add_argument("--mode", choices=["multipart", "s3"], required=True, help="Invocation mode: multipart (local files) or s3 (S3 URIs)")
    parser.add_argument("--loops", type=int, default=DEFAULT_LOOPS, help=f"Total number of requests to send (default: {DEFAULT_LOOPS})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Max concurrent requests (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--output-json-path", default=DEFAULT_OUTPUT_JSON, help=f"Path to write structured JSON results (default: {DEFAULT_OUTPUT_JSON})")
    
    # Optional arguments to override default sample files
    parser.add_argument("--audio-file", help="Path to local audio file for multipart mode (overrides default sample)")
    parser.add_argument("--video-file", help="Path to local video file for multipart mode (overrides default sample, optional)")
    parser.add_argument("--s3-audio-uri", help="S3 URI for audio file for S3 mode (overrides default sample)")
    parser.add_argument("--s3-video-uri", help="S3 URI for video file for S3 mode (overrides default sample, optional)")

    args = parser.parse_args()

    s3_boto_client = boto3.client("s3", region_name=args.region)
    sagemaker_runtime_client = boto3.client("sagemaker-runtime", region_name=args.region)

    # Prepare sample files (download if necessary) ONLY if not using user-provided files
    cached_sample_paths = None
    if (args.mode == "multipart" and not args.audio_file and not args.video_file) or (args.mode == "s3" and not args.s3_audio_uri):
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_sample_paths = prepare_sample_files(s3_boto_client, LOCAL_CACHE_DIR)

    # Determine which sample pairs to use
    sample_pairs_to_use = []
    if args.mode == "multipart":
        if args.audio_file or args.video_file:  # User specified at least one file
            sample_pairs_to_use.append({
                "name": "custom_multipart",
                "audio_path": args.audio_file,
                "video_path": args.video_file
            })
        else:  # Default samples
            sample_pairs_to_use.append({
                "name": "ravdess_multipart",
                "audio_path": str(cached_sample_paths["ravdess_audio"]),
                "video_path": str(cached_sample_paths["ravdess_video"])
            })
            sample_pairs_to_use.append({
                "name": "crema_d_multipart",
                "audio_path": str(cached_sample_paths["crema_d_audio"]),
                "video_path": str(cached_sample_paths["crema_d_video"])
            })
    elif args.mode == "s3":
        if args.s3_audio_uri: # User specified S3 URIs
             sample_pairs_to_use.append({
                "name": "custom_s3",
                "s3_audio_uri": args.s3_audio_uri,
                "s3_video_uri": args.s3_video_uri
            })
        else: # Default S3 URIs
            sample_pairs_to_use.append({
                "name": "ravdess_s3",
                "s3_audio_uri": DEFAULT_SAMPLE_FILES_S3["ravdess_audio"],
                "s3_video_uri": DEFAULT_SAMPLE_FILES_S3["ravdess_video"]
            })
            sample_pairs_to_use.append({
                "name": "crema_d_s3",
                "s3_audio_uri": DEFAULT_SAMPLE_FILES_S3["crema_d_audio"],
                "s3_video_uri": DEFAULT_SAMPLE_FILES_S3["crema_d_video"]
            })
    
    if not sample_pairs_to_use:
        logging.error("No sample files configured for benchmarking. Use defaults or provide custom files/URIs.")
        return

    logging.info(f"Starting benchmark: Endpoint='{args.endpoint}', Region='{args.region}', Mode='{args.mode}', Loops={args.loops}, Workers={args.max_workers}")

    latencies_ms = []
    successful_requests = 0
    failed_requests = 0
    first_error_response = None
    
    overall_start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i in range(args.loops):
            sample_pair = sample_pairs_to_use[i % len(sample_pairs_to_use)] # Cycle through sample pairs
            futures.append(
                executor.submit(
                    run_benchmark_request,
                    sagemaker_runtime_client,
                    args.endpoint,
                    args.region,
                    args.mode,
                    sample_pair
                )
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                latency, success, error_msg = future.result()
                if success:
                    latencies_ms.append(latency)
                    successful_requests += 1
                else:
                    failed_requests += 1
                    if first_error_response is None: # Log first error
                        first_error_response = error_msg or "Unknown error"
                        logging.error(f"First error encountered: {first_error_response}")
                    if error_msg and "Invalid response structure" in error_msg: # Fail fast on bad response
                        logging.error("Aborting run due to invalid response structure.")
                        # Could cancel remaining futures, but for now let them complete or timeout
                        # for f in futures: f.cancel() # This might be too aggressive
                        break 
            except Exception as e:
                logging.error(f"Request generated an exception: {e}")
                failed_requests += 1
                if first_error_response is None:
                    first_error_response = str(e)

    overall_duration_s = time.perf_counter() - overall_start_time

    # --- Metrics Calculation ---
    p50_latency_ms = np.percentile(latencies_ms, 50) if latencies_ms else 0
    p95_latency_ms = np.percentile(latencies_ms, 95) if latencies_ms else 0
    avg_latency_ms = np.mean(latencies_ms) if latencies_ms else 0
    throughput_rps = successful_requests / overall_duration_s if overall_duration_s > 0 else 0

    # --- Human Summary ---
    logging.info("\n--- Benchmark Summary ---")
    logging.info(f"Endpoint:          {args.endpoint}")
    logging.info(f"Region:            {args.region}")
    logging.info(f"Mode:              {args.mode}")
    logging.info(f"Total Requests:    {args.loops}")
    logging.info(f"Concurrent Workers:{args.max_workers}")
    logging.info(f"Successful:        {successful_requests}")
    logging.info(f"Failed:            {failed_requests}")
    if first_error_response:
        logging.info(f"First Error:       {first_error_response[:200]}...") # Truncate long errors
    logging.info(f"Total Duration:    {overall_duration_s:.2f} s")
    if latencies_ms:
        logging.info(f"Avg Latency:       {avg_latency_ms:.2f} ms")
        logging.info(f"P50 Latency:       {p50_latency_ms:.2f} ms")
        logging.info(f"P95 Latency:       {p95_latency_ms:.2f} ms")
    logging.info(f"Throughput:        {throughput_rps:.2f} req/s")
    logging.info("-----------------------\n")

    # --- Structured JSON Output ---
    results_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "endpoint_name": args.endpoint,
        "aws_region": args.region,
        "mode": args.mode,
        "total_requests_scheduled": args.loops,
        "concurrent_workers": args.max_workers,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "first_error_details": first_error_response,
        "total_duration_seconds": round(overall_duration_s, 3),
        "avg_latency_ms": round(avg_latency_ms, 2) if latencies_ms else None,
        "p50_latency_ms": round(p50_latency_ms, 2) if latencies_ms else None,
        "p95_latency_ms": round(p95_latency_ms, 2) if latencies_ms else None,
        "throughput_rps": round(throughput_rps, 2),
        "all_latencies_ms": [round(l, 2) for l in latencies_ms] if latencies_ms else []
    }
    
    output_json_path = pathlib.Path(args.output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    if output_json_path.exists() and output_json_path.stat().st_size > 0:
        try:
            with open(output_json_path, 'r') as f:
                content = f.read()
                if content.strip(): # Ensure content is not just whitespace
                    try:
                        loaded_data = json.loads(content)
                        if isinstance(loaded_data, list):
                            all_results = loaded_data
                        else: # If it was a single object, make it a list
                            all_results.append(loaded_data)
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse existing content of {output_json_path} as JSON. Starting fresh.")
                        all_results = [] # Start fresh if parsing fails
                else: # File exists but is empty or whitespace
                    all_results = []
        except Exception as e:
            logging.warning(f"Error reading or parsing existing {output_json_path}: {e}. Starting fresh.")
            all_results = []
            
    all_results.append(results_data)
    
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"Benchmark results saved to: {output_json_path} ({len(all_results)} entries)")

if __name__ == "__main__":
    main()
