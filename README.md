## Setup and run docker

```
git submodule update --init --recursive
chmod +x run.sh start_mm_tmux.sh
rerun --web-viewer
./run.sh
```
## Run in docker 
```
./start_mm_tmux.sh
```
or manually run
```
# 1. piper startup
bash find_all_can_port.sh
bash can_activate.sh can0 1000000 "1-4.2.2:1.0"
ros2 launch piper start_single_piper.launch.py

# 2. camera startup
ros2 launch piper d435i_high_resolution.launch.py

# 3. action server startup
set -a
source .env
ros2 run mm_actions mm_actions_node
```

## Helpers
```
# set robot joint position to all 0s (home pose)
ros2 topic pub --once /joint_states sensor_msgs/msg/JointState "{name: ['joint1','joint2','joint3','joint4','joint5','joint6','gripper'], position: [0,0,0,0,0,0,0.1]}"

# reset piper (used after pressing green button)
ros2 service call /reset_srv std_srvs/srv/Trigger

# send action: grasp the medicine can
ros2 action send_goal /task_command mm_interface/action/TaskCommand "{command: 'grasp the medicine can'}"

```
