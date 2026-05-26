from setuptools import setup

package_name = 'robot_arm_ik'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Touseef Qamar',
    maintainer_email='touseefqamar1@gmail.com',
    description='Simulation stub satisfying the robot_arm_ik import gate in ik_ws.py.',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
