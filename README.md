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
#### ORB Keypoints
![ORB_features_Stereo.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/ORB_features_Stereo.png)

#### All Matches
![all_matches_Stereo.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/all_matches_Stereo.png)

#### Good Matches
![good_matches_Stereo.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/good_matches_Stereo.png)


## Keypoint Matching by using ORB features for arbitary two frames
```
./build/ORB_KP_opencv
```
#### ORB Keypoints
![ORB_features.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/ORB_features.png)

#### All Matches
![all_matches.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/all_matches.png)

#### Good Matches
![good_matches.png](https://github.com/HugoNip/UndirectDirectSLAM/blob/master/results/good_matches.png)


## Reference
[Source 1](https://github.com/HugoNip/VisualOdometry-KeypointsMatching)  
[Source 2](https://github.com/HugoNip/VisualOdometry-DirectMethod)
