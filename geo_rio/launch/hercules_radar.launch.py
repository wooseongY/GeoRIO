import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = 'geo_rio'
PACKAGE_SHARE_DIRECTORY = get_package_share_directory(PACKAGE_NAME)


def generate_launch_description():
    config_yaml_fusion = os.path.join(
        PACKAGE_SHARE_DIRECTORY,
        'config',
        'config_hercules_parkinglot.yaml')
    config_rviz = os.path.join(
        PACKAGE_SHARE_DIRECTORY,
        'config',
        'config.rviz')
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', config_rviz, '--ros-args', '--log-level', 'fatal']),
        launch_ros.actions.Node(
            package=PACKAGE_NAME,
            executable='GEORIO',
            name='GEORIO',
            emulate_tty=True,
            output='log',
            parameters=[config_yaml_fusion],
            arguments=['--ros-args', '--log-level', 'info']),
        launch_ros.actions.Node(
            package=PACKAGE_NAME,
            executable='Mapping',
            name='Mapping',
            emulate_tty=True,
            output='log',
            parameters=[config_yaml_fusion],
            arguments=['--ros-args', '--log-level', 'info'])
    ])
