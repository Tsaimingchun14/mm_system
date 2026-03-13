FROM ros:humble-ros-base

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-controller-manager \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-joint-state-publisher \
    tmux \
    can-utils \
    ethtool \
    iproute2 \
    ros-${ROS_DISTRO}-realsense2-camera \
    ros-${ROS_DISTRO}-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system python-can piper_sdk 
#catkin-pkg==1.1.0 lark==1.3.1 empy==3.3.4 scipy
COPY main_ws/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["bash"]
