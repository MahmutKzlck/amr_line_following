import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('ac_line_following'),
        'config',
        'line_detection.rviz',
    )

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='line_detection_rviz',
            output='screen',
            additional_env={
                'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
                'ROS_LOCALHOST_ONLY': '0',
            },
            arguments=['-d', rviz_config],
        ),
    ])
