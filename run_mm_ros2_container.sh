#!/bin/bash

IMAGE_NAME="mm_ros2"
CONTAINER_NAME="mm_ros2_container"
DOCKERFILE="Dockerfile.ros2"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .

xhost +local:docker > /dev/null

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${ROOT_DIR}/main_ws:/workspace/main_ws" \
    -v "${ROOT_DIR}/piper_ros:/workspace/piper_ros" \
    -v "/workspace/main_ws/build" \
    -v "/workspace/piper_ros/build" \
    -v "/workspace/main_ws/install" \
    -v "/workspace/piper_ros/install" \
    -v "/workspace/main_ws/log" \
    -v "/workspace/piper_ros/log" \
    "${IMAGE_NAME}"
