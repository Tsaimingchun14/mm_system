#!/bin/bash

IMAGE_NAME="mm_pytorch"
CONTAINER_NAME="mm_pytorch_container"
DOCKERFILE="Dockerfile.pytorch"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

xhost +local:docker > /dev/null

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e DISPLAY=$DISPLAY \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    -v "${ROOT_DIR}/tmp:/tmp" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${ROOT_DIR}/perception:/workspace/perception" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${IMAGE_NAME}"
