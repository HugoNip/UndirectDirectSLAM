# Introduction


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


# Reference
[Source 1](https://github.com/HugoNip/VisualOdometry-KeypointsMatching)  
[Source 2](https://github.com/HugoNip/VisualOdometry-DirectMethod)
