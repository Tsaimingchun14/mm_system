from setuptools import find_packages, setup

package_name = 'joint_planner'

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
    maintainer='mctsai',
    maintainer_email='jerry110030014@gapp.nthu.edu.tw',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qp_servo_piper_node = joint_planner.qp_servo_piper_node:main',
            'qp_servo_node = joint_planner.qp_servo_node:main',
        ],
    },
)
