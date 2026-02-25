from setuptools import find_packages, setup

package_name = 'task_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mctsai',
    maintainer_email='jerry110030014@gapp.nthu.edu.tw',
    description='High-level task planner that maps natural-language commands to '
                'structured motion tasks using the Gemini Robotics-ER API.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_planner_node = task_planner.task_planner_node:main',
        ],
    },
)
