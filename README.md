# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!


[//]: # (Image References)
[heatmaps]: ./output_images/heatmaps.png
[hog]: ./output_images/HOG_example.png
[pipeline_outputs]: ./output_images/pipeline_examples.png
[search_15]: ./output_images/search_scale_1.5.png
[search_all]: ./output_images/search_windows.png
[static_examples]: ./output_images/static_examples.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output_images/project5_output.mp4


---

### Writeup / README

#### 1. README

The code is inside the notebook `project5.ipynb`.

The output video is located at `./output_images/project5_output.mp4`.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to extract HOG features is the `get_hog_features()` function in the notebook.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) with the lessons from Udacity.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(10, 10)` and `cells_per_block=(2, 2)`:


![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

For the color space and color channel, I decide to use all three channels of *YCrCb* color space because *YCrCb* shows the best HOG features (according to classifier accuracy).

I choose `pixels_per_cell=(10, 10)` to reduce the number of features (originally the `pixels_per_cell=(8, 8)`).
I keep `cells_per_block=(2, 2)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Besides HOG features, I also use two more kinds of features: spatial binning of colors (function `bin_spatial()`), and histogram of colors (function `color_hist()`).

Functions `single_img_features()` and `extract_features()` are used to extract features from all images, cars and non-cars.

After having the samples, I normalize the features by using `sklearn.preprocessing.StandardScaler()`. Then I split the samples into training and validation sets.

I train a Linear SVM classifier using default parameters. I have tested various values for the C parameter, but the difference is minimal.

The accuracy on the test set is 0.9904. While the accuracy is not that great, one important thing in car detection is speed, and I must use Linear SVM with reduced number of features to maintain an acceptable speed. In the beginning I have trained a SVM classifier with non-linear kernel and get a high accuracy rate (~0.9970), but the speed is very slow for my computer.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I search across the whole x-axis, and the range [390..700] in y-axis. I use five scales of windows: (64, 64), (96, 96), (128, 128), (192, 192), (256, 256).

For each scale of window, I crop the region of interest then I calculate the HOG gradient values of the whole region before sliding windows. That way avoids repetitively finding the HOG features.

Here are the sliding windows with size (96, 96).

![alt text][search_15]

Here are all the sliding windows (there are a lot of them).

![alt text][search_all]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline detects cars pretty well on the test images. Note that you can see some false positives. We will deal on these in the next section.

![alt text][static_examples]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project5_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The windows that are marked as positive by the classifier are recorded. Then I threshold to find the areas that cover multiple positive windows, and show it as a heatmap. The areas that do not span enough number of windows (default threshold = 1) are disregarded.

I used `scipy.ndimage.measurements.label()` to identify distinct areas in the heatmap. Each distinct area is assumed to correspond to a vehicle. Finally I draw bounding boxes to cover those detected areas.

##### Here are three test images and their corresponding heatmaps:

![alt text][heatmaps]


##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][pipeline_outputs]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are multiple difficulties with this approach to detect cars. First, it is not clear how to find good features for the classifier. More features might lead to a better classifier, but might cause overfitting. Also the more features used the slower the classifier works.

Secondly, sliding window approach is slow because we need to cover the whole frame, and with different window scales. Then it is tricky to avoid false positives. Drawing bounding boxes can be problematic because the windows have fixed size, but the cars vary in size.

Many fellow students in the program have succesfully used a deep learning approach for this assignment. I will try it in the future.
