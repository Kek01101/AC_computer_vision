import numpy as np
import cv2
import matplotlib.pyplot as plt

# Loading camera calibration numbers into respective variables
arr = np.load('cameraCalNums.npz')
ret = arr['arr_0']
mtx = arr['arr_1']
dist = arr['arr_2']
rvecs = arr['arr_3']
tvecs = arr['arr_4']

# Load in frames somewhere here !!
img = cv2.imread('test_images/straight_lines1.jpg')

# Creating optimal camera matrix based upon img shape and undistorting
h, w = img.shape[:2]
newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
img = cv2.undistort(img, mtx, dist, None, newmtx)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

# Applying color thresholding
s_binary = np.zeros_like(s)
s_binary[(s >= 160) & (s <= 255)] = 1
v_binary = np.zeros_like(v)
v_binary[(v >= 206) & (v <= 255)] = 1

# Applying sobel gradients - need to grayscale img first
sobel_min = 30
sobel_max = 140
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel = np.absolute(sobel)
sobel = np.uint8(255 * sobel / np.max(sobel))
sobel_binary = np.zeros_like(sobel)
sobel_binary[(sobel >= sobel_min) & (sobel <= sobel_max)] = 1

# Stacking the channels to differentiate their contributions from each other
binary = np.dstack((sobel_binary, v_binary, s_binary)) * 255

# Combining the binary threshodls
combined_binary = np.zeros_like(sobel_binary)
combined_binary[(s_binary == 1) | (sobel_binary == 1) | (v_binary == 1)] = 1

# Plotting threshold images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title("Stacked thresholds")
ax1.imshow(binary)
ax2.set_title("Combined thresholds")
ax2.imshow(combined_binary, cmap="gray")
plt.show()
