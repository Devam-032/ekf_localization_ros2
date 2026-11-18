#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get share directories for the two packages
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    ekf_loc_dir = get_package_share_directory('ekf_localization')
    
    # RViz config and world are installed with this package
    rviz_config_path = os.path.join(ekf_loc_dir, 'rviz', 'default.rviz')
    world = os.path.join(ekf_loc_dir, 'worlds', 'fyp_world.world')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # Gazebo with the landmark world + TurtleBot3 spawned at the origin (where the EKF starts)
    empty_world_launch = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzserver.launch.py')),
            launch_arguments={'world': world}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_dir, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui'))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_gazebo_dir, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_gazebo_dir, 'launch', 'spawn_turtlebot3.launch.py')),
            launch_arguments={'x_pose': '0.0', 'y_pose': '0.0'}.items()
        ),
    ])
    
    map_reader_node = TimerAction(
        period=0.0,
        actions=[
            Node(
                package='ekf_localization',
                executable='map_reader',
                name='ekf_localization_node',
                output='screen'
            )
        ]
    )

    map_sub_node = TimerAction(
        period=0.0,
        actions=[
            Node(
                package='ekf_localization',
                executable='map_sub',
                name='ekf_localization_node',
                output='screen'
            )
        ]
    )

    # Delay the EKF localization node launch by 7 seconds
    ekf_node = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='ekf_localization',
                executable='localization',
                name='ekf_localization_node',
                output='screen',
                remappings=[('/odom', LaunchConfiguration('odom_topic'))]
            )
        ]
    )
    
    # Delay the RViz launch by 7 seconds as well
    if rviz_config_path:
        rviz_node = TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config_path],
                    output='screen'
                )
            ]
        )
    else:
        rviz_node = TimerAction(
            period=7.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    output='screen'
                )
            ]
        )
    
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('gui', default_value='true',
                                        description='Start the Gazebo client'))
    ld.add_action(DeclareLaunchArgument('odom_topic', default_value='/odom',
                                        description='Odometry the EKF predicts with (e.g. a noisy relay)'))
    ld.add_action(empty_world_launch)
    ld.add_action(map_reader_node)
    ld.add_action(map_sub_node)
    ld.add_action(ekf_node)
    ld.add_action(rviz_node)
    return ld

if __name__ == '__main__':
    generate_launch_description()
