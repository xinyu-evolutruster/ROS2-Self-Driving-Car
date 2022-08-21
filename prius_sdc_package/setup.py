import os
from glob import glob
from setuptools import setup

package_name = 'prius_sdc_package'
config_module = "prius_sdc_package/config"
detection_module = "prius_sdc_package/detection"
detection_lane_module = "prius_sdc_package/detection/lanes"
detection_sign_module = "prius_sdc_package/detection/signs"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, config_module, detection_module, detection_lane_module, detection_sign_module],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recorder_node = prius_sdc_package.video_recorder:main',
            'spawner_node = prius_sdc_package.sdf_spawner:main',
            'computer_vision_node = prius_sdc_package.computer_vision_node:main' 
        ],
    },
)
