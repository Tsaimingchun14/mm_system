from setuptools import find_packages, setup

package_name = 'motion_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/d435i_high_resolution.launch.py',
            'launch/d435i_basic.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mctsai',
    maintainer_email='jerry110030014@gapp.nthu.edu.tw',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "motion_planner_node = motion_planner.motion_planner_node:main"
        ],
    },
)
