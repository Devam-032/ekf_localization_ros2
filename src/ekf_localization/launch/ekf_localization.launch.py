#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Get share directories for the two packages
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    ekf_loc_dir = get_package_share_directory('ekf_localization')
    
    # Determine the RViz configuration file: first try "ekf_localization.rviz", then "default.rviz"
    rviz_config_path = os.path.join(ekf_loc_dir, 'rviz', 'ekf_localization.rviz')
    if not os.path.exists(rviz_config_path):
        rviz_config_path = os.path.join(ekf_loc_dir, 'rviz', 'default.rviz')
        if not os.path.exists(rviz_config_path):
            rviz_config_path = None

    # Include the TurtleBot3 empty world launch file (this will start Gazebo)
    empty_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'fyp_world.launch.py')
        )
    )
    
    # Delay the EKF localization node launch by 5 seconds
    ekf_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ekf_localization',
                executable='localization',
                name='ekf_localization_node',
                output='screen'
            )
        ]
    )
    
    # Delay the RViz launch by 5 seconds as well
    if rviz_config_path:
        rviz_node = TimerAction(
            period=5.0,
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
            period=5.0,
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
    ld.add_action(empty_world_launch)
    ld.add_action(ekf_node)
    ld.add_action(rviz_node)
    return ld

if __name__ == '__main__':
    generate_launch_description()
