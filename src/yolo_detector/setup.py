from setuptools import find_packages, setup
import os 
from glob import glob



package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
        glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sidibe',
    maintainer_email='sidibe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_distance_controller_without_twist = yolo_detector.detector_distance_controller_without_twist:main', 
            'Mog2 = yolo_detector.Mog2:main', 
            'fall_detection_node = yolo_detector.fall_detection_node:main',
            'webcam_fall_detection_node = yolo_detector.webcam_fall_detection_node:main',
            'camera_node = yolo_detector.camera_node:main',
            'projet_fuzzy_track = yolo_detector.projet_fuzzy_track:main',
            'projet_fuzzy = yolo_detector.projet_fuzzy:main',
            'fall_track = yolo_detector.fall_track:main',
            'pvclip_fall_node = yolo_detector.pvclip_fall_node:main',
            'fall_avance = yolo_detector.fall_avance:main',
            'fall_detection_method1_node = yolo_detector.fall_detection_method1_node:main',
            'fall_detection_method2_node = yolo_detector.fall_detection_method2_node:main',
            'fall_detection_method3_node = yolo_detector.fall_detection_method3_node:main',
            'fall_detection_method4_node = yolo_detector.fall_detection_method4_node:main',
            'fall_detection_method5_node = yolo_detector.fall_detection_method5_node:main',













        ],
    },
)
