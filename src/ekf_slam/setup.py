import os
from setuptools import find_packages, setup

package_name = 'ekf_slam'

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
            "ekf_slam = ekf_slam.ekf_slam:main",
            "ekf_slam_2 = ekf_slam.ekf_slam_new_approach:main",
            "ekf_slam_3 = ekf_slam.ekf_slam_v_w:main",
        ],
    },
)

