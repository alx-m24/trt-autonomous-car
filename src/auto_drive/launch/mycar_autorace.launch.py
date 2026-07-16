import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch.actions import TimerAction
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('auto_drive')
    urdf_file = os.path.join(pkg, 'urdf', 'my_car.urdf')
    world_file = os.path.join(pkg, 'worlds', 'my_track.world')
    ekf_config = os.path.join(pkg, 'config', 'ekf.yaml')

    models_dir = os.path.join(pkg, 'worlds')
    gazebo_model_path = models_dir + os.pathsep + os.environ.get('GAZEBO_MODEL_PATH', '')

    gui_arg = DeclareLaunchArgument('gui', default_value='true')

    gazebo_cmd = ['gzserver', '--verbose', world_file,
                  '-s', 'libgazebo_ros_factory.so',
                  '-s', 'libgazebo_ros_init.so']

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_car',
            '-topic', 'robot_description',
            '-x', '0.63',
            '-y', '1.06',
            '-z', '0.02',
            '-Y', '0.0',
            '-spawn_service_timeout', '30.0',
        ],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gui_arg,
        ExecuteProcess(cmd=gazebo_cmd, output='screen'),
        # If gui:=true is passed, also launch gzclient
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('gui'))
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
            output='screen'
        ),
        TimerAction(
            period=8.0,
            actions=[spawn_entity_node]
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config],
        ),
    ])
