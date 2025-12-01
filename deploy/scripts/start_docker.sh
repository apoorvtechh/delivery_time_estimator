#!/bin/bash

exec > /home/ubuntu/start_docker.log 2>&1
echo "=== Starting Docker Deployment ==="

AWS_REGION="eu-north-1"
ACCOUNT_A_ID="513278912561"
REPO="delivery-time-estimator"
IMAGE_TAG="latest"
CONTAINER_NAME="delivery-time-estimator"

echo "Fetching secrets from SSM..."

MLFLOW_URI=$(aws ssm get-parameter --name "/external/mlflow_uri" --query "Parameter.Value" --output text)
AWS_KEY=$(aws ssm get-parameter --name "/external/accountA/aws_access_key_id" --with-decryption --query "Parameter.Value" --output text)
AWS_SECRET=$(aws ssm get-parameter --name "/external/accountA/aws_secret_access_key" --with-decryption --query "Parameter.Value" --output text)


echo "Logging into ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 513278912561.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
docker pull ${ACCOUNT_A_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO}:${IMAGE_TAG}

echo "Stopping old container if exists..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Running NEW container (structure exactly like you want)..."

docker run -d \
  -p 80:8000 \
  -e MLFLOW_TRACKING_URI="$MLFLOW_URI" \
  -e AWS_ACCESS_KEY_ID="$AWS_KEY" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET" \
  --name $CONTAINER_NAME \
  ${ACCOUNT_A_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO}:${IMAGE_TAG}

echo "=== Container started successfully ==="
