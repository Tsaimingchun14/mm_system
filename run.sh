#!/bin/bash

IMAGE_NAME="mm"
CONTAINER_NAME="mm_container"
DOCKERFILE="Dockerfile"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --gpus all \
    --privileged \
    -v "${ROOT_DIR}/main_ws:/workspace/main_ws" \
    -v "${ROOT_DIR}/piper_ros:/workspace/piper_ros" \
    -v "/workspace/main_ws/build" \
    -v "/workspace/piper_ros/build" \
    -v "/workspace/main_ws/install" \
    -v "/workspace/piper_ros/install" \
    -v "/workspace/main_ws/log" \
    -v "/workspace/piper_ros/log" \
    -v "${ROOT_DIR}/start_mm_tmux.sh:/workspace/start_mm_tmux.sh" \
    "${IMAGE_NAME}" 