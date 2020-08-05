# Introduction
This project shows several methods for keypoint matching,
including Single-Level ORB, Multi-Level ORB, Optical Flow based methods, 
and the keypoint matching result in direct method SLAM.

# Requirements
## OpenCV
### Required Packages
OpenCV  
OpenCV Contrib

# Compile this Project
```
mkdir build
cd build
cmake ..
make 
```

# Run
## Keypoint Matching by using ORB features for stereo images
```
./build/ORB_KP_Stereo_opencv
```
### Good Matches
![good_matches_Stereo.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/good_matches_Stereo.png)


## Keypoint Matching by using ORB features for arbitary two frames
### ComputeORB function is defined by OpenCV
```
./build/ORB_KP_opencv
```
### Good Matches
![good_matches.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/good_matches.png)


## Keypoint Matching by using ORB features for arbitary two frames
### ComputeORB function is not defined by OpenCV
```
./build/ORB_KP_function
```
### Good Matches
![matches_function.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/matches_function.png)


## Keypoint Matching by using Optical Flow (Lucas–Kanade method, LK) for arbitary two frames
```
./build/LK_KP
```
### Multi-Level Optical Flow
![LK_Multi.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/LK_Multi.png)


## Keypoint Matching in Direct Method based SLAM
```
./build/directSLAM
```
### Optical Flow
![directSLAM_pointMatching.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/directSLAM_pointMatching.png)

## Keypoint Matching by using Multi-Level ORB features
```
./build/ORB_KP_Multilevel
```
### Multi-Level result
Extract ORB cost = 0.0118106 seconds.  
Match ORB cost = 0.0142241 seconds.  
Matches: 112  

![matches_ORBMultiLayer.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/matches_ORBMultiLayer.png)


### Single-Level result
Extract ORB cost = 0.00274763 seconds.   
Match ORB cost = 0.00102879 seconds.   
Matches: 65  

![matches_ORBSingleLayer.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/matches_ORBSingleLayer.png)


### Partial Scale Invariant

ORB algorithm uses a multiscale image pyramid. An image pyramid is a multiscale representation of a single image, 
that consist of sequences of images all of which are versions of the image at different resolutions. 
Each level in the pyramid contains the downsampled version of the image than the previous level. 
Once orb has created a pyramid it uses the fast algorithm to detect keypoints in the image. 
By detecting keypoints at each level orb is effectively locating key points at a different scale. 
In this way, ORB is partial scale invariant.

![imagepyramids.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/imagepyramids.png)

## SIFT
![figure_1.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/figure_1.png)

## SURF
![figure_2.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/figure_2.png)

# Reference
[Source 1](https://github.com/HugoNip/VisualOdometry-KeypointsMatching)  
[Source 2](https://github.com/HugoNip/VisualOdometry-DirectMethod)  
[Source 3](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf)
