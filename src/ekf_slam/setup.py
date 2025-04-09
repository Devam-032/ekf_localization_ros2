from setuptools import find_packages, setup

package_name = 'ekf_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
