from modulefinder import packagePathMap
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    # package_dir = get_package_share_directory('prius_sdc_package')
    # world_file = os.path.join(package_dir, 'worlds', 'self_driving_car.world')

    return LaunchDescription(
        [
            Node(
                package = 'prius_sdc_package',
                executable = 'recorder_node',
                name = 'recorder',
                output = "screen"
            ),
            Node(
                package = 'teleop_twist_keyboard',
                executable = 'teleop_twist_keyboard',
                name = 'driver',
                output = "screen"
            )
        ]
    )