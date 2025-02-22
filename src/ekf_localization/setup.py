from setuptools import find_packages, setup

package_name = 'ekf_localization'

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
            "test_pub = ekf_localization.pub:main",
            "test_sub = ekf_localization.sub:main",
            "test_pubsub = ekf_localization.pubsub:main",
            "lidar_sub = ekf_localization.lidar_sub:main",
            "odom_sub = ekf_localization.odom_sub:main",
            "localization = ekf_localization.ekf_localization:main",
        ],
    },
)
