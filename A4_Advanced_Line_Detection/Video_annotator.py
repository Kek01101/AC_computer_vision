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

# Creating a histogram of the warped image
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

# Setting up the canvas to draw lines based upon histogram
out_img = np.dstack((warped, warped, warped))*255
midpoint = np.int32(histogram.shape[0]//2)
xleft_base = np.argmax(histogram[:midpoint])
xright_base = np.argmax(histogram[midpoint:]) + midpoint

# Sliding window parameters
needNewWindow = False
left_fit = np.array([-2.71823279e-05, 1.32617022e-02, 3.63059951e+02])
right_fit = np.array([7.99650215e-07, 9.32131259e-03, 9.31589498e+02])
windows = 9
margin = 100
minpix = 50
winheight = np.int32(warped.shape[0]//windows)

# Positions of nonzero pixels
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Drawing new lane lines - could be sliding window or around-poly search
if needNewWindow:
    # If new window needed, it will be drawn here

    # Current window positions
    xleft = xleft_base
    xright = xright_base

    # Lane indicies
    left_ids = []
    right_ids = []

    # Window being drawn
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
else:
    # If new window not needed, searches around old polynomial
    left_ids = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_ids = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]
    left = [nonzerox[np.int32(left_ids)], nonzeroy[np.int32(left_ids)]]
    right = [nonzerox[np.int32(right_ids)], nonzeroy[np.int32(right_ids)]]

# Drawing colored pixels onto out_img and sorting nonzeros into left and right - need to ignore ones not within boxes
out_img[left[1], left[0]] = [255,0,0]
out_img[right[1], right[0]] = [0,0,255]

# Fitting a polynomial onto the pixels from out_img
left_fit = np.polyfit(left[1], left[0], 2)
right_fit = np.polyfit(right[1], right[0], 2)
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
left_line = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_line = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Creating dewarped image
# Converting polynomials into points for cv2 poly
left_win_1 = np.array([np.transpose(np.vstack([left_line-10, ploty]))])
left_win_2 = np.array([np.flipud(np.transpose(np.vstack([left_line+10, ploty])))])
right_win_1 = np.array([np.transpose(np.vstack([right_line-10, ploty]))])
right_win_2 = np.array([np.flipud(np.transpose(np.vstack([right_line+10, ploty])))])
left_line_pts = np.hstack((left_win_1, left_win_2))
right_line_pts = np.hstack((right_win_1, right_win_2))

# Drawing lines onto warped color image and creating a mask of the lines
gray_lines = np.zeros_like(warped)
dewarp = np.dstack((gray_lines, gray_lines, gray_lines))*255
cv2.fillPoly(dewarp, np.int_([left_line_pts]),(0,255,0))
cv2.fillPoly(dewarp, np.int_([right_line_pts]),(0,255,0))
cv2.fillPoly(gray_lines, np.int_([left_line_pts]),(255,255,255))
cv2.fillPoly(gray_lines, np.int_([right_line_pts]),(255,255,255))

# Dewarping lines and mask
dematrix = cv2.getPerspectiveTransform(end_points, start_points)
dewarped = cv2.warpPerspective(dewarp, dematrix, (x,y), flags=cv2.INTER_NEAREST)
gray_lines = cv2.warpPerspective(gray_lines, dematrix, (x,y), flags=cv2.INTER_NEAREST)

# Using mask to overlay lines onto original image
dewarped_1 = cv2.bitwise_and(img,img,mask=cv2.bitwise_not(gray_lines))
dewarped_2 = cv2.bitwise_and(dewarped,dewarped,mask=gray_lines)
dewarped = cv2.add(dewarped_1, dewarped_2)

# Displaying completed image
cv2.imshow("Dewarped?", dewarped)
cv2.waitKey(10000)
cv2.destroyAllWindows()