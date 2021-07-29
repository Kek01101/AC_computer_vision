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

# Applying color thresholding - Indvidually done on both saturation and value
s_binary = np.zeros_like(s)
s_binary[(s >= 160) & (s <= 255)] = 1
v_binary = np.zeros_like(v)
v_binary[(v >= 206) & (v <= 255)] = 1

# Applying sobel gradients - need to grayscale img first
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel = np.absolute(sobel)
sobel = np.uint8(255 * sobel / np.max(sobel))
sobel_binary = np.zeros_like(sobel)
sobel_binary[(sobel >= 30) & (sobel <= 140)] = 1

# Combining the binary thresholds
combined_binary = np.zeros_like(sobel_binary)
combined_binary[(s_binary == 1) | (sobel_binary == 1) | (v_binary == 1)] = 1

# Warping the image into a birds-eye view
x = combined_binary.shape[1]
y = combined_binary.shape[0]
start_points = np.float32([[598, 452],[299, 654],[1020, 645],[715, 452]])
end_points = np.float32([[335, 0],[350, y],[945, y],[955, 0]])
matrix = cv2.getPerspectiveTransform(start_points, end_points)
warped = cv2.warpPerspective(combined_binary, matrix, (x,y), flags=cv2.INTER_NEAREST)

# Plotting threshold images
plt.imshow(warped, cmap="gray")
plt.show()