import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('auto_drive')
    urdf_file = os.path.join(pkg, 'urdf', 'my_car.urdf')
    world_file = os.path.join(pkg, 'worlds', 'my_track.world')
    ekf_config = os.path.join(pkg, 'config', 'ekf.yaml')

    # Make model://track_race resolve to this package's worlds/track_race/.
    # The worlds dir holds the track_race model folder; prepend it to
    # GAZEBO_MODEL_PATH so Gazebo finds it regardless of workspace name.
    models_dir = os.path.join(pkg, 'worlds')
    gazebo_model_path = models_dir + os.pathsep + os.environ.get('GAZEBO_MODEL_PATH', '')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
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
                # On-road spawn for the track_race mesh. The mesh is re-centred
                # (model.sdf pose -1.2132 0.0694), so the origin sits off the road
                # and the car fell through. This point is interior road surface
                # (drivable plane z~=0.003), ~0.4 m clearance from the nearest edge.
                '-x', '0.63',
                '-y', '1.06',
                # Spawn origin == wheel-contact plane, so z is the drop height.
                # Road top is z~=0.003; 0.02 leaves a ~1.7 cm gap so the car
                # settles gently instead of dropping 10 cm.
                '-z', '0.02',
                '-Y', '0.0',
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