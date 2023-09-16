# KDD-LOAM
KDD-LOAM: Jointly Learned Keypoint Detector and Descriptors Assisted LiDAR Odometry and Mapping 


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04 or 20.04 (tested).

ROS Kinetic or Melodic or Noetic (tested). [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL**
PCL 1.12.0 is tested.
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

### 1.3. **FMT** and **Sophus**

Install FMT, the prerequisite of Sophus:
```
    git clone https://github.com/fmtlib/fmt.git
    cd fmt
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
```

Clone the repository and make:

```
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
```

### 1.4. **TBB**


### 1.5. **Python** and **CUDA**, **CUDNN**
Python 3 (Python 3.7 is tested, conda is recommended).

requirements (pip installation): numpy, scipy, torch, torchvision, open3d, rospy

## 2. Build KDD-LOAM
Clone the repository and catkin_make:

```
    mkdir -p ~/kddloam_ws/src
    cd ~/kddloam_ws/src
    git clone https://github.com/NeSC-IV/KDD-LOAM.git
    cd ../
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Compile the C++ extensions to support the neural network:
```
    cd ~/kddloam_ws/src/scripts/cpp_extensions
    sh compile_wrappers.sh
    cd ~/kddloam_ws/TCKDD/cpp_extensions
    sh compile_wrappers.sh
    cd ~/kddloam_ws/TCKDD
    mkdir checkpoints
    mkdir logs
```

## 3. KITTI Example (Velodyne HDL-64)
Download [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to YOUR_DATASET_FOLDER and set the `dataset_folder` and `sequence_number` parameters in `kitti_publisher.launch` file. Note you also convert KITTI dataset to bag file for easy use by setting proper parameters in `kitti_helper.launch`. 

```
    source ~/kddloam_ws/devel/setup.bash
    cd ~/catkin_ws/src/scripts
    python keypointsDescription.py --num_keypoints 5000
    python odometry.py --num_keypoints 4500 --multi_threads_mode True
    rosrun aloam_velodyne kddloam_velodyne_HDL_64.launch
    roslaunch aloam_velodyne kitti_publisher.launch
```

## 4.Acknowledgements
Thanks for LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time, RSS 2014) and [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).

Thanks for [CT-ICP](https://github.com/jedeschaud/ct_icp) and [KISS-ICP](https://github.com/PRBonn/kiss-icp).

Thanks for [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

If you use this library for any academic work, please cite our original paper.
