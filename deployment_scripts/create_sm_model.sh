#!/usr/bin/env bash
# Create SageMaker model, endpoint-config, and endpoint for Fusion-Emotion
# Usage: ./create_sm_model.sh [ECR_IMAGE_URI] [MODEL_NAME] [ENDPOINT_CONFIG] [ENDPOINT_NAME] [INSTANCE_TYPE] [ROLE_ARN] [S3_MODEL_DATA (optional)]
set -euo pipefail

ECR_IMAGE_URI="$1"
MODEL_NAME="$2"
ENDPOINT_CONFIG="$3"
ENDPOINT_NAME="$4"
INSTANCE_TYPE="$5"
ROLE_ARN="$6"
S3_MODEL_DATA="${7:-}"

REGION="${REGION:-us-west-2}"

# Delete existing resources if present
aws sagemaker delete-endpoint --endpoint-name "${ENDPOINT_NAME}" --region "${REGION}" || true
aws sagemaker delete-endpoint-config --endpoint-config-name "${ENDPOINT_CONFIG}" --region "${REGION}" || true
aws sagemaker delete-model --model-name "${MODEL_NAME}" --region "${REGION}" || true

# Create model
if [[ -n "${S3_MODEL_DATA}" ]]; then
  echo "Creating SageMaker model with external model data: ${S3_MODEL_DATA}"
  aws sagemaker create-model \
    --model-name "${MODEL_NAME}" \
    --primary-container Image="${ECR_IMAGE_URI}",ModelDataUrl="${S3_MODEL_DATA}" \
    --execution-role-arn "${ROLE_ARN}" \
    --region "${REGION}" || echo "Model already exists – continuing."
else
  echo "Creating SageMaker model with image only"
  aws sagemaker create-model \
    --model-name "${MODEL_NAME}" \
    --primary-container Image="${ECR_IMAGE_URI}" \
    --execution-role-arn "${ROLE_ARN}" \
    --region "${REGION}" || echo "Model already exists – continuing."
fi

# Create endpoint-config
aws sagemaker create-endpoint-config \
  --endpoint-config-name "${ENDPOINT_CONFIG}" \
  --production-variants VariantName=AllTraffic,ModelName="${MODEL_NAME}",InitialInstanceCount=1,InstanceType="${INSTANCE_TYPE}" \
  --region "${REGION}" || echo "Endpoint-config exists – continuing."

# Create endpoint
aws sagemaker create-endpoint \
  --endpoint-name "${ENDPOINT_NAME}" \
  --endpoint-config-name "${ENDPOINT_CONFIG}" \
  --region "${REGION}" || echo "Endpoint creation in progress or exists."

# Wait for endpoint to be InService
echo "Waiting for endpoint to reach InService status..."
while true; do
  STATUS=$(aws sagemaker describe-endpoint --endpoint-name "${ENDPOINT_NAME}" --region "${REGION}" --query EndpointStatus --output text)
  echo "Current status: $STATUS"
  if [[ "$STATUS" == "InService" ]]; then
    echo "Endpoint is InService."
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    echo "Endpoint deployment failed."
    exit 1
  else
    sleep 30
  fi
done
