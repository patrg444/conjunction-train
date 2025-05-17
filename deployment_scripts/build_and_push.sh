#!/usr/bin/env bash
# Build and push the Fusion-Emotion Docker image to ECR (linux/amd64 only)
# Usage: ./build_and_push.sh [REGION] [ECR_REPO] [IMAGE_TAG]
set -euo pipefail

REGION="${1:-us-west-2}"
ECR_REPO="${2:-fusion-emotion}"
IMAGE_TAG="${3:-latest-$(date +%Y%m%d%H%M%S)}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "Ensuring ECR repo ${ECR_REPO} in ${REGION}..."
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" > /dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}"

echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Building and pushing Docker image to ECR: ${ECR_URI}"
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker buildx build --platform linux/amd64 --provenance=false -f deployment_scripts/Dockerfile -t "${ECR_URI}" --push .

echo "Image pushed: ${ECR_URI}"
