cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_STANDARD 14)


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
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
)

add_library(
  ransac
  include/aloam_velodyne/ransac.h
  src/ransac.cpp
)

add_dependencies(ransac ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ransac ${catkin_LIBRARIES})

link_directories(${Open3D_LIBRARY_DIRS})
link_directories(${OpenMP_LIBRARIES})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)


add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})


add_executable(kittiPublisher src/kittiPublisher.cpp)
target_link_libraries(kittiPublisher ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(curvatureEstimate src/curvatureEstimate.cpp)
target_link_libraries(curvatureEstimate ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(odometry src/odometry.cpp)
target_link_libraries(odometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Open3D_LIBRARIES})

add_executable(mapping src/mapping.cpp)
add_dependencies(mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(mapping ransac ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Open3D_LIBRARIES} ${OpenMP_LIBRARIES})


#catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
