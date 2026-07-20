cmake_minimum_required(VERSION 3.8)
project(auto_drive)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(sensor_listener src/sensor_listener.cpp)
add_executable(manual_control src/manual_control.cpp)
add_executable(occupancy_grid_node src/occupancy_grid_node.cpp)
add_executable(auto_control src/auto_control.cpp)

target_compile_features(sensor_listener PRIVATE cxx_std_17)
target_compile_features(manual_control PRIVATE cxx_std_17)
target_compile_features(occupancy_grid_node PRIVATE cxx_std_17)
target_compile_features(auto_control PRIVATE cxx_std_17)

ament_target_dependencies(sensor_listener
    rclcpp
    nav_msgs
    sensor_msgs
    cv_bridge
    OpenCV
)

ament_target_dependencies(manual_control
    rclcpp
    geometry_msgs
)

ament_target_dependencies(auto_control
    rclcpp
    geometry_msgs
    visualization_msgs
)

ament_target_dependencies(occupancy_grid_node
    rclcpp
    visualization_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    OpenCV
    image_geometry
    tf2
    tf2_ros
    tf2_geometry_msgs
)

target_include_directories(occupancy_grid_node PRIVATE include)
target_include_directories(sensor_listener PRIVATE include)
target_include_directories(manual_control PRIVATE include)
target_include_directories(auto_control PRIVATE include)

install(TARGETS sensor_listener manual_control occupancy_grid_node auto_control
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
