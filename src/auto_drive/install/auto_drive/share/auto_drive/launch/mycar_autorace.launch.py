import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('auto_drive')
    urdf_file = os.path.join(pkg, 'urdf', 'my_car.urdf')
    world_file = os.path.join(pkg, 'worlds', 'my_track.world')
    ekf_config = os.path.join(pkg, 'config', 'ekf.yaml')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_file,
                 '-s', 'libgazebo_ros_factory.so',
                 '-s', 'libgazebo_ros_init.so'],
            output='screen'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
            output='screen'
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'my_car',
                '-topic', 'robot_description',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1',
            ],
            output='screen'
        ),
        # EKF: fuse planar_move /odom + IMU /imu -> /odometry/filtered.
        # Gives the control node a trustworthy yaw to terminate the 90deg
        # dead-reckoned corner maneuver on.
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config],
        ),
    ])