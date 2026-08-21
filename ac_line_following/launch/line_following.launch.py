import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DDS_ENVIRONMENT = {
    'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
    'ROS_LOCALHOST_ONLY': '0',
}


def generate_launch_description():

    package_share_directory = get_package_share_directory(
        'ac_line_following'
    )

    config_file = os.path.join(
        package_share_directory,
        'config',
        'line_following.yaml'
    )

    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='camera',
        name='camera',
        output='screen',
        additional_env=DDS_ENVIRONMENT,
        condition=IfCondition(LaunchConfiguration('start_camera')),
        remappings=[
            (
                '/camera/camera/color/image_raw',
                '/ac_line_following/internal/camera/image_raw',
            ),
            (
                '/camera/camera/color/camera_info',
                '/ac_line_following/internal/camera/camera_info',
            ),
        ],
        parameters=[{
            'enable_color': True,
            'enable_depth': False,
            'enable_infra': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'enable_motion': False,
            'enable_rgbd': False,
            'pointcloud.enable': False,
            'align_depth.enable': False,
            'publish_tf': False,
            'color_qos': 'SENSOR_DATA',
            'color_info_qos': 'SENSOR_DATA',
            'rgb_camera.color_profile': '640x360x30',
            'rgb_camera.color_format': 'RGB8',
            'camera.color.image_raw.enable_pub_plugins': [
                'image_transport/raw',
            ],
        }],
    )

    line_detector_node = Node(
        package='ac_line_following',
        executable='line_detector',
        name='line_detector_node',
        output='screen',
        additional_env=DDS_ENVIRONMENT,
        parameters=[config_file],
    )

    line_controller_node = Node(
        package='ac_line_following',
        executable='line_controller',
        name='line_controller_node',
        output='screen',
        additional_env=DDS_ENVIRONMENT,
        parameters=[config_file],
        condition=IfCondition(LaunchConfiguration('start_controller')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_camera',
            default_value='true',
            description='Start the Intel RealSense color camera node.',
        ),
        DeclareLaunchArgument(
            'start_controller',
            default_value='false',
            description='Start motion controller (disabled during detection tests).',
        ),
        realsense_node,
        line_detector_node,
        line_controller_node,
    ])
