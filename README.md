# KDD-LOAM
KDD-LOAM: Jointly Learned Keypoint Detector and Descriptors Assisted LiDAR Odometry and Mapping 


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04 or 20.04 (tested).
ROS Kinetic or Melodic or Noetic (tested). [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3. **PCL**
PCL 1.12.0 is tested.
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

### 1.4. **Python** and **CUDA**, **CUDNN**
Python 3 (Python 3.7 is tested, conda is recommended).
requirements (pip installation): numpy, scipy, torch, torchvision, nibabel, open3d, rospy

## 2. Build KDD-LOAM
Clone the repository and catkin_make:

```
    cd ~/catkin_ws/src
    git clone https://github.com/NeSC-IV/KDD-LOAM.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
    cd ~/catkin_ws/scripts
    sh compile_wrappers.sh
```

## 3. KITTI Example (Velodyne HDL-64)
Download [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to YOUR_DATASET_FOLDER and set the `dataset_folder` and `sequence_number` parameters in `kitti_helper.launch` file. Note you also convert KITTI dataset to bag file for easy use by setting proper parameters in `kitti_helper.launch`. 

```
    roslaunch aloam_velodyne aloam_velodyne_HDL_64.launch
    roslaunch aloam_velodyne kitti_helper.launch
```

## 4.Acknowledgements
Thanks for LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time) and [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).
