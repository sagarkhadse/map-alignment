cmake_minimum_required(VERSION 3.0.2)
project(map_align)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
)