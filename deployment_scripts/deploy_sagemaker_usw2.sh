#!/usr/bin/env bash
# Automated deployment of the Fusion-Emotion SageMaker endpoint in us-west-2
# ‑ Creates an execution role
# ‑ Builds & pushes Docker image to ECR
# ‑ Creates SageMaker model, endpoint-config, and endpoint
#
# Prerequisites:
#   • AWS CLI v2 configured with creds for account 324037291814
#   • Docker daemon running and able to push to ECR
#   • trust_sagemaker.json present in the same directory
#
# Usage:
#   chmod +x deploy_sagemaker_usw2.sh
#   ./deploy_sagemaker_usw2.sh

set -euo pipefail

### ─────────────────────────────────────────────
### Parameters (override via env vars if desired)
### ─────────────────────────────────────────────
REGION=${REGION:-us-west-2}
ROLE_NAME=${ROLE_NAME:-SageMakerFusionExecutionRole}
ECR_REPO=${ECR_REPO:-fusion-emotion}
IMAGE_TAG=${IMAGE_TAG:-latest-$(date +%Y%m%d%H%M%S)} # Unique tag
MODEL_NAME=${MODEL_NAME:-fusion-emotion}
EP_CONFIG=${EP_CONFIG:-fusion-emotion-config}
EP_NAME=${EP_NAME:-fusion-emotion-endpoint}
INSTANCE_TYPE=${INSTANCE_TYPE:-ml.g4dn.xlarge}
PACKAGE=${PACKAGE:-false}
S3_MODEL_DATA=${S3_MODEL_DATA:-""}
MODEL_PACKAGE_NAME=${MODEL_PACKAGE_NAME:-fusion-emotion-mp}
FRAMEWORK=${FRAMEWORK:-PYTORCH}
FRAMEWORK_VERSION=${FRAMEWORK_VERSION:-2.1}

### ─────────────────────────────────────────────
echo "Retrieving AWS account ID …"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
echo "Account: $ACCOUNT_ID"

### ─────────────────────────────────────────────
### 0. Clean up existing SageMaker resources (if any)
### ─────────────────────────────────────────────
echo "Attempting to delete existing SageMaker endpoint ${EP_NAME} (if it exists)..."
aws sagemaker delete-endpoint --endpoint-name "${EP_NAME}" --region "${REGION}" || echo "Endpoint ${EP_NAME} not found or already deleted."

echo "Attempting to delete existing SageMaker endpoint configuration ${EP_CONFIG} (if it exists)..."
aws sagemaker delete-endpoint-config --endpoint-config-name "${EP_CONFIG}" --region "${REGION}" || echo "Endpoint configuration ${EP_CONFIG} not found or already deleted."

echo "Attempting to delete existing SageMaker model ${MODEL_NAME} (if it exists)..."
aws sagemaker delete-model --model-name "${MODEL_NAME}" --region "${REGION}" || echo "Model ${MODEL_NAME} not found or already deleted."

### ─────────────────────────────────────────────
### 1. Create/ensure IAM SageMaker execution role
### ─────────────────────────────────────────────
echo "Checking for IAM role: ${ROLE_NAME}"
if ! aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1; then
  echo "Creating role ${ROLE_NAME}"
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file://$(dirname "$0")/trust_sagemaker.json
else
  echo "Role already exists."
fi

# Attach required policies (idempotent)
for POLICY_ARN in \
  arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly \
  arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
do
  aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${POLICY_ARN}" || true
done

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)
echo "Execution role ARN: ${ROLE_ARN}"

### ─────────────────────────────────────────────
### 2. Build & push Docker image to ECR
### ─────────────────────────────────────────────
echo "Building and pushing Docker image to ECR using helper script..."
./deployment_scripts/build_and_push.sh "${REGION}" "${ECR_REPO}" "${IMAGE_TAG}"

### ─────────────────────────────────────────────
### 3. Create SageMaker model, endpoint-config, and endpoint
### ─────────────────────────────────────────────
echo "Creating SageMaker model, endpoint-config, and endpoint using helper script..."
if [[ -n "${S3_MODEL_DATA}" && "${S3_MODEL_DATA}" != "\"\"" ]]; then
  ./deployment_scripts/create_sm_model.sh "${ECR_URI}" "${MODEL_NAME}" "${EP_CONFIG}" "${EP_NAME}" "${INSTANCE_TYPE}" "${ROLE_ARN}" "${S3_MODEL_DATA}"
else
  ./deployment_scripts/create_sm_model.sh "${ECR_URI}" "${MODEL_NAME}" "${EP_CONFIG}" "${EP_NAME}" "${INSTANCE_TYPE}" "${ROLE_ARN}"
fi

### ─────────────────────────────────────────────
### 4. Optionally publish Model Package for Marketplace
### ─────────────────────────────────────────────
if [[ "${PACKAGE}" == "true" ]]; then
  if [[ -z "${S3_MODEL_DATA}" || "${S3_MODEL_DATA}" == "\"\"" ]]; then
    echo "S3_MODEL_DATA must be set to the S3 URI of the model weights for Marketplace packaging."
    exit 1
  fi
  echo "Publishing Model Package for Marketplace..."
  ./deployment_scripts/publish_model_package.sh "${ECR_URI}" "${S3_MODEL_DATA}" "${MODEL_PACKAGE_NAME}" "${FRAMEWORK}" "${FRAMEWORK_VERSION}" "${REGION}"
fi

echo "Deployment script finished."
echo "Check endpoint status with:"
echo "  aws sagemaker describe-endpoint --endpoint-name ${EP_NAME} --region ${REGION} --query EndpointStatus"
