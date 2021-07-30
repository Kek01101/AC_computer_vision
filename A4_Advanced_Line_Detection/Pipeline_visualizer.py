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
cv2.imshow("Undistorted image", img)
cv2.imwrite("output_images/Undistored_img.png", img)
cv2.waitKey(1000)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

# Applying color thresholding - Indvidually done on both saturation and value
s_binary = np.zeros_like(s)
s_binary[(s >= 160) & (s <= 255)] = 1
plt.imshow(s_binary, cmap="gray")
plt.show()
v_binary = np.zeros_like(v)
v_binary[(v >= 206) & (v <= 255)] = 1
plt.imshow(v_binary, cmap="gray")
plt.show()

# Applying sobel gradients - need to grayscale img first
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel = np.absolute(sobel)
sobel = np.uint8(255 * sobel / np.max(sobel))
sobel_binary = np.zeros_like(sobel)
sobel_binary[(sobel >= 30) & (sobel <= 140)] = 1
plt.imshow(sobel_binary, cmap="gray")
plt.show()

# Combining the binary thresholds
combined_binary = np.zeros_like(sobel_binary)
combined_binary[(s_binary == 1) | (sobel_binary == 1) | (v_binary == 1)] = 1
plt.imshow(combined_binary)
plt.show()

# Warping the image into a birds-eye view
x = combined_binary.shape[1]
y = combined_binary.shape[0]
start_points = np.float32([[598, 452],[299, 654],[1020, 645],[715, 452]])
end_points = np.float32([[335, 0],[350, y],[945, y],[955, 0]])
matrix = cv2.getPerspectiveTransform(start_points, end_points)
warped = cv2.warpPerspective(combined_binary, matrix, (x,y), flags=cv2.INTER_NEAREST)
plt.imshow(warped)
plt.show()

# Creating a histogram of the warped image
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

# Setting up the canvas to draw lines based upon histogram
out_img = np.dstack((warped, warped, warped))*255
midpoint = np.int32(histogram.shape[0]//2)
xleft_base = np.argmax(histogram[:midpoint])
xright_base = np.argmax(histogram[midpoint:]) + midpoint

# Sliding window parameters
windows = 9
margin = 100
minpix = 50
winheight = np.int32(warped.shape[0]//windows)

# Positions of nonzero pixels
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Current window positions
xleft= xleft_base
xright = xright_base

# Lane indicies
left_ids = []
right_ids = []

for window in range(windows):
    # Window boundaries
    y_low = warped.shape[0] - (window+1)*winheight
    y_high = warped.shape[0] - window*winheight
    xleft_low = xleft - margin
    xleft_high = xleft + margin
    xright_low = xright - margin
    xright_high = xright + margin

    # Draw windows on visualization
    cv2.rectangle(out_img,(xright_low, y_low),(xright_high, y_high),(0,255,0),2)
    cv2.rectangle(out_img,(xleft_low, y_low),(xleft_high, y_high),(0,255,0),2)

    # Find nonzeroes within the window
    win_left_ids = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                    (nonzerox >= xleft_low) & (nonzerox < xleft_high)).nonzero()[0]
    win_right_ids = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                     (nonzerox >= xright_low) & (nonzerox < xright_high)).nonzero()[0]
    left_ids = np.append(left_ids, win_left_ids)
    right_ids = np.append(right_ids, win_right_ids)

    # Recenter windows if need be
    if len(win_left_ids) > minpix:
        leftx = np.int32(np.mean(nonzerox[win_left_ids]))
    if len(win_right_ids) > minpix:
        rightx = np.int32(np.mean(nonzeroy[win_right_ids]))

    # Concatenate the list of indices
    try:
        left_ids = np.concatenate(left_ids)
        right_ids = np.concatenate(right_ids)
    except ValueError:
        pass

    # Package up pixel positions into co-ords
    left = [nonzerox[np.int32(left_ids)], nonzeroy[np.int32(left_ids)]]
    right = [nonzerox[np.int32(right_ids)], nonzeroy[np.int32(right_ids)]]

# Drawing colored pixels onto out_img and sorting nonzeros into left and right - need to ignore ones not within boxes
out_img[left[1], left[0]] = [255,0,0]
out_img[right[1], right[0]] = [0,0,255]
plt.imshow(out_img)
plt.show()

# Fitting a polynomial onto the pixels from out_img
left_fit = np.polyfit(left[1], left[0], 2)
right_fit = np.polyfit(right[1], right[0], 2)
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
left_fit = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plotting final images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.imshow(out_img)
plt.plot(left_fit, ploty, color='purple')
plt.plot(right_fit, ploty, color='purple')
plt.show()