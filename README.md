## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
<img src="writeup_images/main.jpg" width="600px">

[This](https://github.com/udacity/CarND-Advanced-Lane-Lines) Udacity's repository contains starting files for the Project.

My detailed solution **[writeup](https://github.com/feklistoff/udacity-carnd-project4/blob/master/Writeup_Project_4.md)**.

### The Project
---

The steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Files:
* `advanced_lane_find_final.ipynb` - Final pipeline class.
* `building_pipeline.ipynb` - Building pipeline steps and visualization.
* `calibration_parameters.ipynb` - Camera calibraton steps.
* `project_video.mp4` , `challenge_video.mp4`, `harder_video.mp4` - Processed videos
* `coeffs.p` - Saved calibration parameters

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

or Python 3.5 and the following libraries installed:

* [Jupyter](http://jupyter.org/)
* [NumPy](http://www.numpy.org/)
* [Open CV](http://opencv.org/)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
