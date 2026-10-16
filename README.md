TRT Grid-Based Autonomous Car

A grid-based autonomous driving project developed with Taylor's Racing Team (TRT) at Taylor's University, Malaysia.

This project explores autonomous vehicle navigation through a simulated racing environment, combining computer vision, road perception, path generation, and vehicle control. Built on ROS 2 and Gazebo, the system processes camera input to identify the drivable road and generate a path for the vehicle to follow.

Repository: "trt-autonomous-car" (https://github.com/alx-m24/trt-autonomous-car)

Portfolio: "Alexandre Mohamudally" (https://alx-portfolio.pages.dev/)

---

Overview

Autonomous driving requires a vehicle to interpret its surroundings, determine a suitable path, and continuously adjust its movement to remain on course.

This project brings these components together in a simulated racing environment. The vehicle uses camera-based perception to identify the road, estimates its centreline, and communicates path information to a separate control system.

The project is developed as part of the Taylor's Racing Team autonomous vehicle initiative, with an emphasis on perception, navigation, and integrating software components into a working autonomous driving system.

Key Features

- Camera-based perception: Processes simulated camera images using OpenCV to identify the road and its boundaries.
- Bird's-eye-view transformation: Transforms camera imagery into a top-down representation to simplify road geometry analysis.
- Path generation: Extracts road centreline information and publishes a path for the vehicle to follow.
- Temporal filtering: Uses a Kalman filter and smoothing techniques to reduce noise and improve perception stability.
- Corner detection: Analyses road geometry to identify upcoming sharp turns and estimate their direction.
- Modular architecture: Separates perception and vehicle control into ROS 2 nodes that communicate through topics.
- Simulation environment: Uses Gazebo Classic to test autonomous driving behaviour in a controlled racing environment.

System Architecture

The autonomous driving pipeline is divided into distinct stages.

         Gazebo Simulation
                |
                v
        Simulated Camera
                |
                v
       Image Preprocessing
                |
                v
       Road / Lane Detection
                |
                v
      Centreline Estimation
                |
                v
      Path and Corner Output
                |
                v
         Vehicle Control
                |
                v
       Steering and Movement
                |
                +------> Simulation

1. Perception

The perception node receives camera images and processes them using OpenCV and NumPy. Perspective transformation and image segmentation help distinguish the drivable road from its surroundings.

The detected road geometry is then used to estimate a centreline and construct a path in the vehicle's reference frame.

2. Path Generation

The perception system converts image-space observations into path information that can be consumed by the controller. Filtering and continuity checks help reduce unstable path estimates caused by noisy images, gaps, or misleading road segments.

3. Vehicle Control

A separate control node consumes the generated path and determines how the vehicle should move. This separation allows perception and control to be developed and tested independently.

4. Simulation

Gazebo Classic provides the simulated vehicle, camera, and racing environment. ROS 2 connects the simulated sensors to the perception and control components.

Technology Stack

Technology| Purpose
ROS 2 Humble| Robotics middleware and node communication
Python| Perception and autonomous driving logic
OpenCV| Image processing and road detection
NumPy| Numerical operations and filtering
Gazebo Classic 11| Vehicle and environment simulation
Robot Localization| State estimation and sensor fusion infrastructure

Project Structure

The repository contains the ROS 2 workspace and the autonomous driving package.

trt-autonomous-car/
├── .gazebo/
├── src/
│   └── auto_drive/
│       ├── auto_drive/
│       │   ├── perception_node.py
│       │   └── ...
│       └── ...
├── REPLICATE_WORKSPACE.md
└── README.md

The "auto_drive" package contains the autonomous driving nodes and related ROS 2 resources. The workspace replication guide documents the required environment, dependencies, and simulation setup.

Getting Started

Prerequisites

The documented development environment targets:

- Ubuntu 22.04 LTS
- ROS 2 Humble
- Python 3.10
- Gazebo Classic 11
- OpenCV and NumPy
- ROS 2 build tools

1. Install dependencies

Install ROS 2 Humble and the required system packages:

sudo apt update

sudo apt install -y \
  ros-humble-desktop \
  ros-dev-tools \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-localization \
  ros-humble-cv-bridge \
  ros-humble-turtlebot3-gazebo \
  python3-colcon-common-extensions

Install the Python dependencies:

pip3 install \
  "numpy==1.21.5" \
  "opencv-python==4.11.0.86" \
  "opencv-contrib-python==4.13.0.92"

Source the ROS 2 environment:

source /opt/ros/humble/setup.bash

2. Clone the repository

git clone https://github.com/alx-m24/trt-autonomous-car.git
cd trt-autonomous-car

3. Prepare the Gazebo environment

The custom racing track model must also be installed in Gazebo's model directory.

The repository's ""REPLICATE_WORKSPACE.md"" (REPLICATE_WORKSPACE.md) explains how to transfer the "track_race" model into:

~/.gazebo/models/track_race/

Without this external model, Gazebo may be unable to load the intended racing environment.

4. Build the workspace

From the workspace root:

source /opt/ros/humble/setup.bash

colcon build --packages-select auto_drive

source install/setup.bash

5. Launch the simulation

Start the simulation and state estimation:

ros2 launch auto_drive mycar_autorace.launch.py

In separate terminals, with the ROS 2 and workspace environments sourced, start perception and control:

ros2 run auto_drive perception_node

ros2 run auto_drive control_node

For additional configuration, troubleshooting, and recommended perception parameters, refer to ""REPLICATE_WORKSPACE.md"" (REPLICATE_WORKSPACE.md).

Development Challenges

Autonomous navigation involves more than detecting a road in a single image. The system must produce sufficiently stable path estimates for the controller to react consistently.

Important engineering challenges include:

- Perception stability: Reducing fluctuations in detected road geometry across consecutive frames.
- Path continuity: Preventing the generated path from jumping to disconnected road segments or unrelated regions.
- Corner handling: Distinguishing sharp turns from gradual curves and determining when a dedicated turning manoeuvre is needed.
- Coordinate calibration: Converting image measurements into meaningful spatial coordinates for navigation.
- System integration: Ensuring that camera input, perception, state estimation, control, and simulation work together through ROS 2.

These challenges make the project an opportunity to explore the interaction between computer vision, robotics middleware, and autonomous vehicle control.

My Contribution

I contributed to the development of the grid-based autonomous car project for Taylor's Racing Team (TRT).

The project forms part of my broader interest in systems engineering, simulation, and the software that connects algorithms to physical behaviour.

For the specific components I worked on, see the repository's commit history and source code.

Future Improvements

Potential areas for further development include:

- Improving navigation reliability across more complex track layouts.
- Refining grid-based path representation and route selection.
- Improving obstacle detection and avoidance.
- Evaluating path-tracking accuracy and control stability quantitatively.
- Testing the perception and control pipeline beyond the simulated environment.

Acknowledgements

Developed in collaboration with members of Taylor's Racing Team (TRT) at Taylor's University, Malaysia.

This repository documents my contribution to the team's autonomous vehicle development work.

---

Author: "Alexandre Mohamudally" (https://github.com/alx-m24)

GitHub: "alx-m24/trt-autonomous-car" (https://github.com/alx-m24/trt-autonomous-car)

Portfolio: "alx-portfolio.pages.dev" (https://alx-portfolio.pages.dev/)