from glob import glob
from setuptools import find_packages, setup

package_name = 'auto_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/my_car.urdf', 'urdf/my_car.sdf']),
        ('share/' + package_name + '/worlds', ['worlds/my_track.world']),
        # track_race mesh model (travels with the package so model://track_race
        # resolves via GAZEBO_MODEL_PATH set in the launch file).
        ('share/' + package_name + '/worlds/track_race',
            ['worlds/track_race/model.config', 'worlds/track_race/model.sdf']),
        ('share/' + package_name + '/worlds/track_race/meshes',
            glob('worlds/track_race/meshes/*.stl')),
        # Materials were MISSING from the install (fixed 2026-07-03): Gazebo
        # resolves model://track_race to the INSTALL dir, so without these the
        # Track/CustomRoad script can't load and the road renders WHITE ->
        # perception's dark-road mask is empty (road_conf=0, no overlays).
        ('share/' + package_name + '/worlds/track_race/materials/scripts',
            glob('worlds/track_race/materials/scripts/*.material')),
        ('share/' + package_name + '/worlds/track_race/materials/textures',
            glob('worlds/track_race/materials/textures/*.png')),
        # old_track: optional flat-box texture track. Installed so it appears in
        # Gazebo's Insert panel (worlds/ is on GAZEBO_MODEL_PATH) and can be
        # dragged into the running sim on demand.
        ('share/' + package_name + '/worlds/old_track',
            ['worlds/old_track/model.config', 'worlds/old_track/model.sdf']),
        ('share/' + package_name + '/worlds/old_track/materials/scripts',
            glob('worlds/old_track/materials/scripts/*.material')),
        ('share/' + package_name + '/worlds/old_track/materials/textures',
            glob('worlds/old_track/materials/textures/*.png')),
        ('share/' + package_name + '/launch', ['launch/mycar_autorace.launch.py']),
        ('share/' + package_name + '/config', ['config/ekf.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jackerost',
    maintainer_email='brianchong333@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = auto_drive.perception_node:main',
            'control_node = auto_drive.control_node:main',
            'parking_detector_node = auto_drive.parking_detector_node:main',
            'wheel_odom_node = auto_drive.wheel_odom_node:main',
            'terrain_node = auto_drive.terrain_node:main',
            'calibrate_perspective = auto_drive.calibrate_perspective:main',
        ],
    },
)