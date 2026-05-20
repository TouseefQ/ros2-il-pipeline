from setuptools import setup, find_packages

package_name = 'il_webserver_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'teleop_bridge_node = il_webserver_bridge.teleop_bridge_node:main',
            'webserver_action_node = il_webserver_bridge.webserver_action_node:main',
            'panda_ik_ws_node = il_webserver_bridge.panda_ik_ws_node:main',
        ],
    },
)
