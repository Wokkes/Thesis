cmake_minimum_required(VERSION 2.8.3)
project(publisher)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  camera_info_manager
  sensor_msgs
)


find_package(OpenCV)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(publisher src/main.cpp)
target_link_libraries(publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

