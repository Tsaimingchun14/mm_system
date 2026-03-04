# mm_ws — Mobile Manipulator ROS 2 Workspace

## Create environment
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip sync requirements.txt --cache-dir {CACHE_DIR} 
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 
#optional
export HF_HOME={CACHE_DIR}
```

---

## Package overview

```
task_planner   ←  natural-language commands  →  Gemini Robotics-ER API
     ↓ /motion_planner_command (MotionPlannerCommand)
motion_planner ←  perception (SAM3) + IK planning
     ↓ /joint_planner_command (JointPlannerCommand)
joint_planner ← QP whole-body servo (arm + mobile base)
     ↓ /joint_commands (JointState)
```

---

## Task Planner  (NEW)

The task planner receives natural-language robot commands, uses the
**Gemini Robotics-ER 1.5** model to visually ground the target object in
the camera image, and publishes a structured action payload to `/motion_planner_command`.

### Run
```bash
# Provide your Google AI API key
export GOOGLE_API_KEY=<your-key>

ros2 run task_planner task_planner_node \
  --ros-args \
  -p gemini_api_key:=$GOOGLE_API_KEY \
  -p image_topic:=/image_raw \
  -p command_topic:=/task_command \
  -p motion_planner_command_topic:=/motion_planner_command \
  -p motion_planner_feedback_topic:=/motion_planner_feedback
```

### Send a command
```bash
ros2 topic pub --once /task_command std_msgs/msg/String \
  '{"data": "grasp the red cup"}'
```

The node will:
1. Capture the latest `/image_raw` frame.
2. Query Gemini Robotics-ER with the image + your command.
3. Publish a `MotionPlannerCommand` to `/motion_planner_command`.

---

## Motion Planner

```bash
ros2 run motion_planner motion_planner_node

# Manual task injection (bypasses task_planner)
ros2 topic pub /motion_planner_command mm_interface/msg/MotionPlannerCommand \
  '{request_id: "demo-1", action: "grasp", point: [640.0, 360.0], label: "cup"}'
```

---

## Joint Planner

```bash
# Arm-only (Piper standalone)
ros2 run joint_planner qp_servo_piper_node \
  --ros-args --remap /joint_commands:=/joint_states

# Whole-body (Kachaka + Piper)
ros2 run joint_planner qp_servo_node

# Manual joint planner command (for testing)
ros2 topic pub /joint_planner_command mm_interface/msg/JointPlannerCommand \
  "{request_id: 'demo-1', target_pose: {position: {x: 0.2, y: 0.1, z: 0.2}, orientation: {w: 0.0, x: 0.0, y: 1.0, z: 0.0}}, gripper_width: 0.08}"
```
