cmake_minimum_required(VERSION 3.5)
project(geo_rio)

SET(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17 -O3 -march=native -pthread -fopenmp -Wall -DEIGEN_INITIALIZE_MATRICES_BY_NAN")

add_compile_options(-Wno-reorder) # Disable warnings for reordering of initializers
add_compile_options(-Wno-unused-parameter) # Disable warnings for unused parameters
add_compile_options(-Wno-sign-compare) # Disable warnings for signed/unsigned comparison
add_compile_options(-Wno-unused-but-set-variable)
add_compile_options(-Wno-unused-variable)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(livox_interfaces REQUIRED)
find_package(livox_ros_driver REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(estimate_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_lint_auto REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenMP REQUIRED)
include(FindOpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_library(${PROJECT_NAME}
    src/Mapping.cpp
	src/GEORIO.cpp
)
ament_lint_auto_find_test_dependencies()
include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
)
add_executable(Mapping src/Mapping.cpp)
add_executable(GEORIO src/GEORIO.cpp include/ikd-Tree/ikd_Tree.cpp include/voxelmapper/voxelmapper.cpp)
	ament_target_dependencies(${PROJECT_NAME} 
	rclcpp
	std_msgs
	geometry_msgs
	nav_msgs
	sensor_msgs
	visualization_msgs
	std_srvs
	livox_interfaces
	livox_ros_driver
	livox_ros_driver2
	estimate_msgs
	rosbag2_cpp
	rosbag2_storage
	pcl_conversions
	ament_lint_auto
	tf2_ros
)
target_link_libraries(GEORIO
	${PROJECT_NAME}
	${PCL_LIBRARIES}
	${YAML_CPP_LIBRARIES}
	${OpenMP_LIBS}
)
target_link_libraries(Mapping
	${PROJECT_NAME}
	${PCL_LIBRARIES}
	${YAML_CPP_LIBRARIES}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(TARGETS GEORIO
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS Mapping
  DESTINATION lib/${PROJECT_NAME})
ament_export_dependencies(rclcpp yaml-cpp message_runtime livox_ros_driver livox_ros_driver2 livox_interfaces)
ament_export_libraries(${PROJECT_NAME})
ament_package()
