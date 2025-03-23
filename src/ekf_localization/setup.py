import os
from setuptools import find_packages, setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install resource file for ament_index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        # Install all launch files (make sure you have a 'launch' folder in your package root)
        ('share/' + package_name + '/launch', 
            [os.path.join('launch', f) for f in os.listdir('launch') if f.endswith('.launch.py')])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devam',
    maintainer_email='devamshah82@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test_pub = ekf_localization.pub:main",
            "test_sub = ekf_localization.sub:main",
            "test_pubsub = ekf_localization.pubsub:main",
            "lidar_sub = ekf_localization.lidar_sub:main",
            "odom_sub = ekf_localization.odom_sub:main",
            "localization = ekf_localization.ekf_localization:main",
            "map_reader = ekf_localization.map_reader:main",
            "map_sub = ekf_localization.map_sub:main",
            "map_rot = ekf_localization.map_vis_transform:main",
        ],
    },
)
