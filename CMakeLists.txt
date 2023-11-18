cmake_minimum_required(VERSION 2.8.3)
project(kddloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_STANDARD 17)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
set(FMT_LIBRARIES fmt::fmt)

include_directories(
  include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  #${CERES_INCLUDE_DIRS}
  #${OpenCV_INCLUDE_DIRS}
  #${OpenMP_INCLUDE_DIRS}
)

add_library(
  HybridICP STATIC
  include/kddloam_velodyne/preprocessing.hpp
  include/kddloam_velodyne/AdaptiveThreshold.hpp
  include/kddloam_velodyne/registration.hpp
  include/kddloam_velodyne/VoxelHashMap.hpp
  src/VoxelHashMap.cpp
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)


add_executable(kittiPublisher src/kittiPublisher.cpp)
target_link_libraries(kittiPublisher ${catkin_LIBRARIES} ${PCL_LIBRARIES}) # ${OpenCV_LIBS}

add_executable(voxelGridSampling src/voxelGridSampling.cpp)
target_link_libraries(voxelGridSampling ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(local_mapping src/local_mapping.cpp)
target_link_libraries(local_mapping HybridICP ${catkin_LIBRARIES} ${PCL_LIBRARIES} Sophus::Sophus TBB::tbb fmt::fmt)

add_executable(global_mapping src/global_mapping.cpp)
target_link_libraries(global_mapping HybridICP ${catkin_LIBRARIES} ${PCL_LIBRARIES} Sophus::Sophus TBB::tbb fmt::fmt)

add_executable(laserRender src/laserRender.cpp)
target_link_libraries(laserRender ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(savePath src/savePath.cpp)
target_link_libraries(savePath ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
