cmake_minimum_required(VERSION 3.8)
project(auto_drive)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

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

add_executable(sensor_listener src/sensor_listener.cpp)
add_executable(manual_control src/manual_control.cpp)
add_executable(occupancy_grid_node src/occupancy_grid_node.cpp)

# Map compilation and link-time library targets
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

install(TARGETS sensor_listener manual_control occupancy_grid_node
  DESTINATION lib/${PROJECT_NAME}
)

# Crucial: Install launch files, URDFs, worlds, and params to the share folder
install(DIRECTORY launch config urdf worlds
  DESTINATION share/${PROJECT_NAME}
)


# Copying compile_commands to SRC/build for NVIM
add_custom_command(TARGET manual_control POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/build"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/build/compile_commands.json"
    COMMENT "Copying compile_commands.json to source directory for Neovim LSP"
)

ament_package()
