from setuptools import find_packages, setup

package_name = 'uam_marshalling'

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
    maintainer='jhlee98',
    maintainer_email='jhleee1214@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: BSD 3-Clause',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='UAM Marshalling Recognition and Create Guidance Command',
    license='BSD 3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'marshalling = uam_marshalling.main:main'
        ],
    },
)