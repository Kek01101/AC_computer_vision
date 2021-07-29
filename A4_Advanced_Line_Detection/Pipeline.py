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

# Applying color thresholding
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
hls = hls[:,:,2]
hls_min = 170
hls_max = 255
hls_binary = np.zeros_like(hls)
hls_binary[(hls >= hls_min) & (hls <= hls_max)] = 1

# Applying sobel gradients - need to grayscale img first
sobel_min = 20
sobel_max = 100
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel = np.absolute(sobel)
sobel = np.uint8(255 * sobel / np.max(sobel))
sobel_binary = np.zeros_like(sobel)
sobel_binary[(sobel >= sobel_min) & (sobel <= sobel_max)] = 1

# Stacking the channels to differentiate their contributions from each other
binary = np.dstack((np.zeros_like(sobel_binary), sobel_binary, hls_binary)) * 255

# Combining the binary threshodls
combined_binary = np.zeros_like(sobel_binary)
combined_binary[(hls_binary == 1) | (sobel_binary == 1)] = 1

# Plotting threshold images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title("Stacked thresholds")
ax1.imshow(binary)
ax2.set_title("Combined thresholds")
ax2.imshow(combined_binary, cmap="gray")
plt.show()