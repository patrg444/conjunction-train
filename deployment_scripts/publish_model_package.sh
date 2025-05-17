#!/usr/bin/env bash
# Publish a SageMaker Model Package for AWS Marketplace
# Usage: ./publish_model_package.sh [ECR_IMAGE_URI] [S3_MODEL_DATA] [MODEL_PACKAGE_NAME] [FRAMEWORK] [FRAMEWORK_VERSION] [REGION]
set -euo pipefail

ECR_IMAGE_URI="$1"
S3_MODEL_DATA="$2"
MODEL_PACKAGE_NAME="${3:-fusion-emotion-mp}"
FRAMEWORK="${4:-PYTORCH}"
FRAMEWORK_VERSION="${5:-2.1}"
REGION="${6:-us-west-2}"

MODEL_PACKAGE_DEF="model_package_def.json"

cat > "${MODEL_PACKAGE_DEF}" <<EOF
{
  "ModelPackageName": "${MODEL_PACKAGE_NAME}",
  "InferenceSpecification": {
    "Containers": [
      {
        "Image": "${ECR_IMAGE_URI}",
        "ModelDataUrl": "${S3_MODEL_DATA}",
        "Framework": "${FRAMEWORK}",
        "FrameworkVersion": "${FRAMEWORK_VERSION}",
        "NearestModelName": "resnet-152",
        "Environment": {
          "SAGEMAKER_PROGRAM": "inference_server.py",
          "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
        }
      }
    ],
    "SupportedContentTypes": ["multipart/form-data"],
    "SupportedResponseMIMETypes": ["application/json"],
    "SupportedRealtimeInferenceInstanceTypes": ["ml.g4dn.xlarge", "ml.g5.xlarge"],
    "SupportedTransformInstanceTypes": []
  },
  "CertifyForMarketplace": true
}
EOF

echo "Registering SageMaker Model Package for Marketplace..."
aws sagemaker create-model-package \
  --model-package-name "${MODEL_PACKAGE_NAME}" \
  --cli-input-json file://"${MODEL_PACKAGE_DEF}" \
  --region "${REGION}"

echo "Model Package registered. To retrieve the ARN, run:"
echo "aws sagemaker list-model-packages --model-package-name ${MODEL_PACKAGE_NAME} --region ${REGION} --query 'ModelPackageSummaryList[0].ModelPackageArn'"
