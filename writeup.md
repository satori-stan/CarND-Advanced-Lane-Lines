## Writeup Template


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1.png "Road Transformed"
[image3]: ./output_images/warped.png "Warped Example"
[image4]: ./output_images/binary.png "Binary Example"
[image5]: ./output_images/binary_with_windows.png "Points found in all test images"
[image6]: ./output_images/masked_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This file is the writeup and the code and results can be viewed in the "AdvancedLaneLines.ipynb" notebook file.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The images used to calibrate come out of the provided 20 examples from "./camera_cal", although since the first image ("./camera_cal/calibration1.png") doesn't fully show all calibration corners, it is skipped to be used as an output test image.

Once each image is read, it is converted to grayscale before using the "findChessboardCorners" function to correlate the object points to the image points.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Chessboard (distorted and undistorted) from camera calibration.][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the correction matrix and distortion coefficients are calculated, they can be applied to a test image. The correction is not very noticeable except on the bottom corners of the image, where some of the hood of the car recedes. In the image, it can also be seen the source points for the perspective transformation.

![Test image source (left), undistorted (center) and with source points for perspective transformation (right).][image2]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I chose to warp the image before applying thresholds to the image because it seemed to produce less noise after the image was processed.

The source points for the perspective transform were selected by visually trying to frame the outside of the lane in an image of a section of the road with straight lines. The source points are defined in the third cell.

```python
src = np.array([[[245,680], [1060,680], [689,450], [591,450]]], dtype=np.float32)
```

The destination points transform the trapezoid into a rectangle, shrinking the bottom corners and separating the top corners and keeping the center of the shape in the image center. They are defined in the sixth cell.

With these definitions, we get the transformation matrix in the same cell using the OpenCV getPerspectiveTransform function.

```python
dst = np.array([[[300,720], [980,720], [980,0] ,[300,0]]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src, dst)
```

I hardcoded both the source and destination points because using a different camera that produces a different image size or a camera at a different position (yielding a different source perspective) would require different source and destination points that I am unsure can be dynamically calculated.

For reference, these are the points used formatted as a table (from the bottom left point in clockwise direction):

| Source     | Destination | 
|:----------:|:-----------:| 
| 245, 680   | 300, 720    | 
| 591, 450   | 300, 0      |
| 689, 450   | 980, 0      |
| 1060, 680  | 980, 720    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![The first and second straight line example images warped with and without the destination points drawn.][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (done step-by-step in cell 8). First, a new test image is selected, undistorted and warped (using linear interpolation). A 3 pixel Gaussian blur is applied to get rid of sharp edges that could cause noise in the line detection. Next, the color map is changed to HLS.

The L channel is used twofold: the Sobel gradient on the X axis and a combination of the gradient magnitude and direction. The sobel gradient uses a 15 pixel kernel size for both X and Y gradients. For the X gradient binary, only points between 50 and 100 are kept. For the magnitude binary, only points over 15 are kept. For the orientation binary, only points that are below PI/4 or over 3\*PI/4 are kept. To combine the gradient's magnitude and direction, both conditions must be met.

The S channel is used in at least one and at most two ways: the Sobel gradient on the X axis and the amount of saturation. The sobel gradient is calculated exactly like the X Sobel gradient for the L channel (15 pixel kernel, keeping points between 50 and 100). The S channel first undergoes an exposure rescaling operation to make the image clearer. Then, the saturation binary keeps all points over 200 in value. Furthermore, the saturation binary is only combined with the rest if the binary images if the mean value of the rescaled image is less than 15 units over the median. This threshold was selected as a dynamic value to keep as much information as possible (from selecting the S channel binary) without inserting too much noise. 

Here's an example of my output for this step.

![Binary images for X Sobel, gradient magnitude and the combination of magnitude and orientation for both L and S channels; the S channel binary and the full combination.][image4]

The code to process the individual channels was fit into the "findEdges" function in cell 9, while the whole preprocessing logic of the image (from undistorting to generating the binary image) is compiled in the "preprocess" function in cell 10.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once the binary image is available, the points to fit the lines are found using the sliding window technique. Cell 12 has the logic to apply the window. A difference with the code from the lecture is that only half (left/right) of the image is processed each time, instead of all at once. This is done to avoid having points from a solid line affecting the search of points for a dashed line (usually the left line affecting the search for the right line in the default project video).

![Areas in the binary images found to contain lane points.][image5]

To actually fit a polynomial to the points found, a class "Line" is used. The class is defined in cell 13 of the notebook. In order to perform the calculations correctly, the shape of the space where the line is to be found is required. The first value of the shape defines the "Y" space while the second value is used to calculate the distance to the center of the shape.

To find a line in a frame, the points delimited by the search windows must be passed to the "find" method of the class. If there are actually any points in the variable passed to the method, they are stored for reference and used directly as inputs to the "polyfit" function from NumPy for a second degree curve. A number of useful variables are kept in the class, like the number of frames that have been skipped because the input variables didn't have enough points for a regression, the array of points last used for the regression, a number of polynomial fit values to use as history, an average of the last n fit values, etc.

The usage of the class is demonstrated in cell 14 where a test image is preprocessed, split, the points found by sliding window and the line calculated using the "left" and "right" instances of the Line class.

In the actual pipeline (as demonstrated in cells 18 and 19), the sliding windows are only used if the number of skipped frames is over 5. In all other cases, the pre-existing line is used to find candidate points with a margin of +/- 80 pixels.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the main method of the Line class, the last step in finding a line is to calculate the curvature and distance to center. This is done in a private method of the class and calculated using the point of the line closest to the car as reference for both calculations. In this step, the pixel X and Y coordinates are converted to their real-world counterparts using values that are calculated visually as number of meters per pixel from knowing the distance between lanes and road stretch length.

For the radius of the curvature, the first and second coefficients are used as part of the first and second derivative of the curve required for the calculation.

For the distance to the center, with the y coordinate of the point closest to the car, the x coordinate value is calculated with the quadratic formula (which is the type of function we are fitting the points to). Once the value of the x coordinate in meters is known, the absolute value of the subtraction of the x coordinate and the central position of the image (also in meters) is used. This provides the distance from the line to the center of the car. Adding both left and right distances should yield the lane width. The distance of the car to the center of the lane would then be the distance of a single (left) line to the car minus half the width of the lane.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once the quadratic coefficients are calculated for each line, each x coordinate is calculated for every y coordinate. With this, we now can draw both lines in an empty canvas as shown in cell 15. An inverse perspective transform can then be applied and our new canvas stacked with other layers to produce the desired color and lastly, merged with the undistorted image (cell 16).

![Each of the test images with the lane detection mask applied and displaying the radius of the curvature and distance to the center of the image.][image6]

In these test images, some of the effect of noise can be observed since the lane doesn't follow all the lines.

The above transformations are joined in the "postprocess" function defined in cell 17.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4). It is obtained by merging all the steps (pre-process, find lines, post-process) into a single "process_image" function (cell 19) that can be called by the "fl_image" method of a VideFileClip class of moviepy. Since these images are a sequence, we can take full advantage of the historical values stored by the Line class and a very good result is obtained.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As it is, the pipeline is not perfect, the defects are quickly identified when run against the challenge videos. Additional lines in the same direction of car travel (as with the challenge video) can trip up the thresholding and hence the sliding window algorithm. If any of the lines disappears from view (as with the harder challenge), the logic to find the points to fit the line and draw the lane will fail. It is also possible to have two curves in view (as in the harder challenge) which would require at least a third degree polynomial to fit correctly.

One of the main questions I had and saw asked in the forums was whether it is best to apply the thresholds or to warp the image first. I don't think there is a definitive answer. Applying the thresholds first may cause less noise since warping the image introduces artifacts from the stretching. Warping first may produce a cleaner fit by reducing the distortion on the thresholded points. I think that how to come up with more dynamic thresholds is actually a more productive discussion: changing light conditions, different materials on the road surface, etc. all affect the pipeline result and I don't believe static thresholds can get the best result.

Another interesting point is how to apply and use validations amongst the lines programmatically. For instance: come up with a measure of certainty for a line, if one of the point clouds for a line is not above the desired certainty level, use the coefficients for a line that does and apply an offset. If no lane line is visible, use the edge of the road. This would be the way a person would do it.

Finally, the approach I take to find the point clouds for fitting the line (splitting the source image in left and right before applying the sliding window) is useful for straight stretches of road and slight curves, but breaks in the presence of tight curves and narrower roads. I would have to amend this part of my pipeline to successfully pass the harder challenge.

