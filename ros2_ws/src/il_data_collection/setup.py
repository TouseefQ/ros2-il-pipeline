from setuptools import setup, find_packages

package_name = 'il_data_collection'

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
            'data_collector_node = il_data_collection.data_collector_node:main',
            'dataset_manager_node = il_data_collection.dataset_manager_node:main',
        ],
    },
)
