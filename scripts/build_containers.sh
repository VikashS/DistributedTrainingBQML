#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
VERSION=${VERSION:-$(git rev-parse --short HEAD)}
ARTIFACT_REGISTRY="us-docker.pkg.dev/${PROJECT_ID}/fraud-detection"

echo "Building containers for version ${VERSION}"

# Authenticate with GCP Artifact Registry
gcloud auth configure-docker us-docker.pkg.dev

# Build base image
docker build -t "${ARTIFACT_REGISTRY}/base:${VERSION}" \
  -f containers/base/Dockerfile .

# Build ETL image
docker build -t "${ARTIFACT_REGISTRY}/etl:${VERSION}" \
  --build-arg BASE_IMAGE="${ARTIFACT_REGISTRY}/base:${VERSION}" \
  -f containers/etl/Dockerfile .

# Build standard training image
docker build -t "${ARTIFACT_REGISTRY}/training:${VERSION}" \
  --build-arg BASE_IMAGE="${ARTIFACT_REGISTRY}/base:${VERSION}" \
  -f containers/training/Dockerfile .

# Build XGBoost GPU training image
docker build -t "${ARTIFACT_REGISTRY}/training-xgboost:${VERSION}" \
  --build-arg BASE_IMAGE="${ARTIFACT_REGISTRY}/base:${VERSION}" \
  -f containers/training/xgboost.Dockerfile .

# Build prediction image
docker build -t "${ARTIFACT_REGISTRY}/prediction:${VERSION}" \
  --build-arg BASE_IMAGE="${ARTIFACT_REGISTRY}/base:${VERSION}" \
  -f containers/prediction/Dockerfile .

# Push all images
docker push "${ARTIFACT_REGISTRY}/base:${VERSION}"
docker push "${ARTIFACT_REGISTRY}/etl:${VERSION}"
docker push "${ARTIFACT_REGISTRY}/training:${VERSION}"
docker push "${ARTIFACT_REGISTRY}/training-xgboost:${VERSION}"
docker push "${ARTIFACT_REGISTRY}/prediction:${VERSION}"

echo "Successfully built and pushed all containers"