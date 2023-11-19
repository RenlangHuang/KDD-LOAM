## KDD-LOAM: Jointly Learned Keypoint Detector and Descriptors Assisted LiDAR Odometry and Mapping
This repository represents the official implementation of the paper:

#### [KDD-LOAM: Jointly Learned Keypoint Detector and Descriptors Assisted LiDAR Odometry and Mapping](https://arxiv.org/abs/2309.15394) (ICRA 2024, under review)

[Renlang Huang](https://github.com/RenlangHuang), [Minglei Zhao](https://github.com/2019lelexia), Jiming Chen, [Liang Li](https://github.com/liangli1990) | Zhejiang University

<div align="center">
  <img src=figures/local_map.png width=80%/>
</div>

## 1. Installation
### 1.1. **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04 or 20.04 (tested).

ROS Kinetic or Melodic or Noetic (tested). [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL**
* Run these commands to install dependencies first:
```
sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev -y
sudo apt-get install cmake -y
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev -y
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common -y
sudo apt-get install libflann1.9 libflann-dev -y
sudo apt-get install libeigen3-dev -y
sudo apt-get install libboost-all-dev -y
sudo apt-get install libvtk7.1p-qt libvtk7.1p libvtk7-qt-dev -y
sudo apt-get install libqhull* libgtest-dev -y
sudo apt-get install freeglut3-dev pkg-config -y
sudo apt-get install libxmu-dev libxi-dev -y
sudo apt-get install mono-complete -y
sudo apt-get install openjdk-8-jdk openjdk-8-jre -y
```           
* Then follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html). PCL 1.12.0 has been tested.

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

Follow [oneTBB installation](https://github.com/oneapi-src/oneTBB/blob/master/INSTALL.md). Release tagged 2021.8.0 is tested.

### 1.5. **Python** and **CUDA**, **CUDNN**
Python 3 (Python 3.7 is tested, conda is recommended).

requirements (pip installation): numpy, scipy, torch, torchvision, open3d, rospy

Code has been tested with Ubuntu 20.04, GCC 9.3.0, Python 3.7, PyTorch 1.7.1, CUDA 11.1 and cuDNN 8.1.0.

## 2. Build KDD-LOAM
Clone the repository and catkin_make:

```
mkdir -p ~/kddloam_ws/src
cd ~/kddloam_ws/src
git clone https://github.com/NeSC-IV/KDD-LOAM.git
cd ../
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

Compile the C++ extensions for the neural network:
```
cd ~/kddloam_ws/src/scripts/cpp_extensions
sh compile_wrappers.sh
cd ~/kddloam_ws/src/TCKDD/cpp_extensions
sh compile_wrappers.sh
cd ~/kddloam_ws/src/TCKDD
mkdir checkpoints
mkdir logs
```

### Pre-trained Weights

We provide pre-trained weights in the [release](https://github.com/NeSC-IV/KDD-LOAM/releases/tag/KITTI_KPFCNN) page. Please download the latest weights and put them in the `checkpoints` directory.

## 3. Datasets Preparation

### 3.1. 3DMatch

The dataset can be downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator) to `YOUR_DATA_PATH`, which you are supposed to modify in `/TCKDD/datasets/match3d.py`. The data should be organized as follows:

```text
--YOUR_DATA_PATH--3dmatch--metadata
                        |--data--train--7-scenes-chess--cloud_bin_0.pth
                              |      |               |--...
                              |      |--...
                              |--test--7-scenes-redkitchen--cloud_bin_0.pth
                                    |                    |--...
                                    |--...
```

### 3.2. KITTI Odometry

Download the data from the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to `YOUR_DATA_PATH`, which you are supposed to modify in `/TCKDD/datasets/kitti.py` and `/launch/kitti_publisher.launch`. The data should be organized as follows:

```text
--YOUR_DATA_PATH--KITTI_data_odometry--results (ps: ground truth)
                                    |--sequences--00--calib.txt
                                    |          |  |--times.txt
                                    |          |--...
                                    |--velodyne--sequences--00--velodyne--000000.bin
                                              |              |         |--...
                                              |              |--...
                                              |--pairs (ps: will be generated by our code)
```
## 4. Training, Evaluation, and Inference
To train a model on the indoor 3DMatch dataset, please run:
```
cd TCKDD
python train_3dmatch.py
```
To evaluate a model on the 3DMatch dataset, please run:
```
cd TCKDD
python evaluate_3dmatch.py
```
And you can evaluate various types of keypoint detectors with different number of keypoints of a specified model:
```
python evaluate_3dmatch.py --keypoint_detector prob --num_keypoints 500 --model_path ./checkpoints/3dmatch_kpfcnn_HCL64_40000.pth
```
To detect the 3D keypoints of a point cloud and register it to a reference point cloud via feature matching, you can refer to `inference_3dmatch.py` as an example:
```
python inference_3dmatch.py --keypoint_detector nms --num_keypoints 500 --sample_index 1000 --model_path ./checkpoints/3dmatch_kpfcnn_HCL64_40000.pth
```
Here are some examples of point clouds rendered according to the predicted saliency uncertainty (saliency descends by red->green->blue->purple, i.e., red regions are salient):
<div align="center">
  <img src=figures/prob.png width=31%/><img src=figures/nms.png width=32%/><img src=figures/nms-prob.png width=31%/>
</div>
<div align="center">
  <font>(a) probabilistic detector&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</font>
  <font>(b) NMS detector&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</font>
  <font>(c) probabilistic NMS detector</font>
</div>
<div align="center">
  <img src=figures/indoor-1.png width=29%/><img src=figures/indoor-2.png width=31%/><img src=figures/indoor-3.png width=38%/>
</div>

To train a model on the outdoor KITTI dataset, please run:
```
cd TCKDD
python train_kitti.py
```
To evaluate a model on the KITTI dataset, please run:
```
cd TCKDD
python evaluate_kitti.py
```
And you can evaluate various types of keypoint detectors with different number of keypoints of a specified model:
```
python evaluate_kitti.py --keypoint_detector prob --num_keypoints 5000
```
To detect the 3D keypoints of a point cloud and register it to a reference point cloud via feature matching, you can refer to `inference_3dmatch.py` as an example:
```
python inference_kitti.py --num_keypoints 5000 --sample_index 59 --model_path ./checkpoints/kitti_HCL64_augm_23500.pth
```
<div align="center">
  <img src=figures/kitti.png width=50%/>
</div>

## 5. Odometry and Mapping: KITTI Example (Velodyne HDL-64)
Download [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to `YOUR_DATA_PATH` and set the `dataset_folder` and `sequence_number` parameters in `kitti_publisher.launch` file. You can start the LiDAR odometry and mapping by running the following commands in different terminals to launch the corresponding groups of ROS nodes.
```
python keypointsDescription.py
python odometry.py
roslaunch kddloam_velodyne kddloam.launch
roslaunch kddloam_velodyne kitti_publisher.launch
```

Meanwhile, you can launch the `savePath` roscpp node (refer to `kitti_publisher.launch`) to record the localization results to a txt file, then you can evaluate the relative pose errors through the official [KITTI odometry evaluation tools](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool) after synchronizing the localization results and the ground-truth poses. Note that the performance reported in our paper is evaluated on the premise that no LiDAR frames are discarded during system operation.

### Citation
If you use this library for any academic work, please cite our original paper [KDD-LOAM](http://arxiv.org/abs/2309.15394) (under review, ICRA 2024).

### Acknowledgments
In this project we use (parts of) the official implementations of the following works: 

- [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) (backbone)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch) (data preprocessing)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator) (dataset management)
- [CT-ICP](https://github.com/jedeschaud/ct_icp) (voxel hash map implementation)
- [KISS-ICP](https://github.com/PRBonn/kiss-icp) (motion compensation, data association, robust registration)
- [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) (kitti publisher, scan-to-scan and scan-to-map pipeline)
- [tsl robin-map library](https://github.com/Tessil/robin-map) (from which the `include/tsl` is forked directly)

We thank the respective authors for open sourcing their methods. We would also like to thank reviewers.
