## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### List the files in the repo: notebook, output images, output video

[//]: # (Image References)
[heatmaps]: ./output_images/heatmaps.png
[hog]: ./output_images/HOG_example.png
[pipeline_outputs]: ./output_images/pipeline_examples.png
[search_15]: ./output_images/search_scale_1.5.png
[search_all]: ./output_images/search_windows.png
[static_examples]: ./output_images/static_examples.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output_images/project5_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

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

The pipeline detects cars pretty well on the test images. Note that you can see some false positives. We will deal on these in the next section

![alt text][static_examples]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project5_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heatmaps]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][pipeline_outputs]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
