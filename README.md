**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Note: My code includes the code from Project 4, Advance Lane Lines, please look for the header '##Vehicle Detection Project --- Starts Here' for the start of my code for this project.

[//]: # (Image References)
[image1]: ./output_images/CarNotCar.png
[image2]: ./output_images/hog_examples/hog/hog.jpg
[image3]: ./output_images/search_windows.png
[image4]: ./output_images/boxes1.png
[image42]: ./output_images/boxes2.png
[image43]: ./output_images/boxes3.png
[image44]: ./output_images/boxes4.png
[image45]: ./output_images/boxes5.png
[image46]: ./output_images/boxes6.png


[image5]: ./output_images/heatmap1.png
[image52]: ./output_images/heatmap2.png
[image53]: ./output_images/heatmap3.png
[image54]: ./output_images/heatmap4.png
[image55]: ./output_images/heatmap5.png
[image56]: ./output_images/heatmap6.png



[video1]: ./output_images/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. How HOG features were extracted from the training images.

I have used the code from the quiz, the code that extracts HOG features is located in the IPython notebook, it is called `get_hog_features`  

First I read all the `vehicle` and `not-vehicle` images. This is an example of these images:

![alt text][image1]

I experimented with different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used different images from each of the two classes and displayed them to understand how the `skimage.hog()` was working.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. How I settled final choice of HOG parameters.

I first visualized the different images in the 3-D plot using different color spaces, although the course instructor suggested that vehicles seems to have a high saturation relative to the background, I was not able to make the same observation. I took the advice and used a color space that could identify on saturation, namely HSV, but my results were bad, as the classifier was giving me false positive on the road. I changed to `YCrCb`, which started to give better results.  I then used different values for orientation, from 9 to 18, using a higher number was costing more CPU time and I saw the number of detected boxes went down. I tried many different values and settled on the following based on best performance:
`orientations=9`, `pixels_per_cell=(8, 8)`, `hog_channel=ALL` and `cells_per_block=(2, 2)`   

####3. How I trained SVM classifier using the selected HOG features

I trained a linear SVM see `svc = LinearSVC()`, it uses `extract_features` function, which also adds the `spatial` and `color histgotram` features.

###Sliding Window Search

####1. Sliding window search

As it was discussed in the lectures, the car that is near appears to be larger than the ones that are far away. Therefore a good idea is to use sliding windows of smaller size near the horizon and bigger ones near the bottom side of the image.  I really liked the window selection done by: https://github.com/jimwinquist/vehicle-detection/blob/master/vehicle_detection.py, I copied the window selection logic from there.
Previously I tried somewhat different window sizes, but my results were not as good as the above. In future I will research how can I improve upon the window selection.

The following image is from the same place that I borrowed the window selection logic: https://github.com/jimwinquist/vehicle-detection I used the same image, since it exactly shows how the window selection is being done.

![alt text][image3]

####2. Examples of test images 

I am using `YCrCb` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

Following is an example of how my code detects the vehicles.

![alt text][image4]
![alt text][image42]
![alt text][image43]
![alt text][image44]
![alt text][image45]
![alt text][image46]
---

### Video Implementation

####1. 
Here's a link to my [video1]


####2. 
In each frame of the video I collected the positions of the detected boxes, from these boxes I created a heatmap and then thresholded the heatmap to find out the vehicle positions.

I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, each corresponding to detected vehicles. I then constructed bounding boxes to cover the area of each blob detected.  

Following shows the final detected boxes and the corresponding heat maps.

Note: the titles of the following figures are incorrect, `original` depicts the final-detected vehicles and `processed` the heat maps.
### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]



---

###Discussion

####1. Problems / issues faced during this project  
Where will your pipeline likely fail?  

Future Improvements:
* It would be a good idea to extrapolate the direction of movement and look in that direction for object detection.
* The current pipeline is running very slow, I will look into making it performant. 
* I am using SVM Linear classifier, I would research into using Deep Learning models in order to improve the results.
* Run the pipeline along with the Advance Lane's pipeline.  I have added the code along side the Advanced Lanes code, so that I can run them in in combined pipeline.

