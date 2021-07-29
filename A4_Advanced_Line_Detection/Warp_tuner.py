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

# Warping the image for a bird's eye view
x = img.shape[1]
y = img.shape[0]
start_points = np.float32([[598, 452],[299, 654],[1020, 645],[715, 452]])
end_points = np.float32([[335, 0],[350, y],[945, y],[955, 0]])
matrix = cv2.getPerspectiveTransform(start_points, end_points)
warped = cv2.warpPerspective(img, matrix, (x,y), flags=cv2.INTER_NEAREST)

# Output plotting
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.set_title("Normal image")
ax1.imshow(img)
ax1.axline((598, 452), (299, 654))
ax1.axline((1020, 645), (715, 452))
ax2.set_title("Warped image")
ax2.imshow(warped)
ax2.axline((350, 0), (350, y))
ax2.axline((950, y), (950, 0))
plt.show()