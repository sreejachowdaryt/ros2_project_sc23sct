# ROS2 Project - Autonomous Robot Navigation and Colour Detection

**Student:** Sreeja Chowdary Tulluru
**Module:** COMP3631 - Robotics and Autonomous Systems  
**University of Leeds**

---

## Project Overview

This project implements an autonomous robot system using ROS2 and the TurtleBot3 platform in a Gazebo simulation environment. The robot autonomously explores an environment, detects three RGB coloured boxes (red, green, blue) using computer vision, and navigates to stop within 1 metre of the blue box.

---

## Task Description

The robot is required to:
1. Start from the bottom right compartment of the task world
2. Explore the environment using Nav2 motion planning
3. Detect all three RGB coloured boxes (red, green, blue) using camera-based colour detection
4. Once all three colours are seen and the blue box is visible, autonomously navigate toward the blue box
5. Stop within approximately 1 metre of the centre of the blue box

---

## Project Structure
```bash
ros2_project_sc23sct/
├── map/
│   ├── map.pgm                        # Map image file of the task world
│   └── map.yaml                       # Map metadata (resolution, origin)
├── resource/
│   └── ros2_project_sc23sct           # Package resource marker file
├── ros2_project_sc23sct/
│   ├── __init__.py                    # Package initialiser
│   └── project_node.py                # Main autonomous robot node
├── test/
│   ├── test_copyright.py              
│   ├── test_flake8.py                
│   └── test_pep257.py                 
├── package.xml                        # ROS2 package manifest
├── README.md                          # Project documentation
├── setup.cfg                          # Package setup configuration
└── setup.py                           # Package entry points
```

---

## Implementation

### State Machine

The robot operates using a three-state machine:

| State | Description |
|---|---|
| `EXPLORE` | Navigate through waypoints using Nav2 motion planning |
| `APPROACH_BLUE` | Drive toward blue box using camera feedback |
| `DONE` | Task complete — robot stopped near blue box |

### Motion Planning

The robot uses **Nav2** with the provided map for autonomous navigation. Waypoints were selected heuristically from the map using the RViz Publish Point tool:

- **Waypoint 1** `(-2.11, -6.11)` — Exits the starting compartment, robot sees red and green boxes
- **Waypoint 2** `(-5.84, -9.01)` — Moves robot toward the blue box area

Nav2 automatically plans a collision-free path to each waypoint using the map.

### Colour Detection

Colour detection uses OpenCV with HSV colour space:

| Colour | HSV Hue Centre | OpenCV Range |
|---|---|---|
| Blue | 110 | 100–120 |
| Green | 60 | 50–70 |
| Red | 0/180 | 0–10 and 170–180 |

For each colour:
- Binary mask created using `cv2.inRange()`
- Largest contour found using `cv2.findContours()`
- Bounding box drawn using `cv2.boundingRect()`
- Centre of mass calculated using `cv2.moments()`

### Blue Box Approach

Once all three colours are detected and the current waypoint is complete, the robot switches to camera-based control:

- **Steering:** Error between blue contour centre and image centre → angular velocity
- **Distance:** Contour area used as proxy for distance (larger = closer)
- **Stop condition:** Area ≥ 50,000 pixels ≈ 1 metre from blue box

---

## How to Run

### Prerequisites

Make sure you are in the Singularity environment:
```bash
ros
cd ~/ros2_ws
source ~/.bashrc
```

### Step 1 — Launch Gazebo World
```bash
ros2 launch turtlebot3_gazebo turtlebot3_task_world_2026.launch.py
```

### Step 2 — Launch Nav2
```bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  use_sim_time:=True \
  map:=~/ros2_ws/src/ros2_project_sc23sct/map/map.yaml
```

### Step 3 — Launch RViz and Set Initial Pose
```bash
rviz2
```
In RViz click **2D Pose Estimate** and set the robot's starting position in the bottom right compartment.

### Step 4 — Build and Run the Node
```bash
cd ~/ros2_ws
colcon build --packages-select ros2_project_sc23sct
source install/setup.bash
ros2 run ros2_project_sc23sct project_node
```

---

## Dependencies

- ROS2 Humble
- TurtleBot3 packages
- Nav2 navigation stack
- OpenCV (`cv2`)
- NumPy
- `cv_bridge`

---

## Labs Applied

| Lab | Topic | Applied in Project |
|---|---|---|
| Lab 1 | ROS2 environment setup | Package structure and build system |
| Lab 2 | Publishers and Subscribers | Camera subscriber, cmd_vel publisher |
| Lab 3 | Robot actuation with Twist | Blue box approach control |
| Lab 4 | Mapping and Nav2 navigation | Waypoint navigation with provided map |
| Lab 5 | Computer vision with OpenCV | RGB colour detection with bounding boxes |

---

## License

Developed for academic purposes - COMP3631, University of Leeds, 2025/26.

---