cmake_minimum_required(VERSION 2.8.3)
project(persoonsdetectie)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  camera_info_manager
  sensor_msgs
  message_generation
)

add_message_files(
	FILES
	personen.msg
)

generate_messages(
	DEPENDENCIES
	geometry_msgs
)

find_package(OpenCV)

catkin_package(
	CATKIN_DEPENDS geometry_msgs roscpp std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(persoonsdetectie src/main.cpp)

add_dependencies(persoonsdetectie ${persoonsdetectie_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(persoonsdetectie ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

