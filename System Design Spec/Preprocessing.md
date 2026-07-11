Rotate all images 90 degrees so that they are portrait mode (rather than current landscape mode).
Based on the videos, crop until the millisecond on the iPad is same, and calculate the corresponding frames for all other data points (odometry, IMU, confidence, depth, rgb) and delete everything before that.
## Offset angle
First, determine which depth is useable. We dispose of all depth coordinates that have confidence coordinate value 0. Now, if there exists a continuous region of depth values that are confidence 2 and the region is at least 1 ninth of the image, calculate directly using those depths as suggested below. Else, incorporate confidence 1 points in the calculation.
Calculate, using depth, for the angle in which the camera is angled at with respect to the vertical. Using the camera matrix and focal length with cv2 in python, distort the RGB back to reality. 

Based on the size of the barbell circle, scale all videos such that the barbell circle is same size. Trim all images using odometry till the cropped images are the same size and represent same region.

 