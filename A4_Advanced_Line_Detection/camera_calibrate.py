import numpy as np
import cv2
import glob

# Object points?
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)

# Object and image point arrays
objpoints = []
imgpoints = []

# Loading in all images
images = glob.glob('camera_cal/*.jpg')

# Takes all loaded images and processes them
for cal_image in images:
    # Importing image and converting to grayscale
    img = cv2.imread(cal_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Finding chess board corners
    ret, corners = cv2.findChessboardCorners(img, (9, 6), None)

    # If there are chess board corners, save image and object points to array
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)


cv2.destroyAllWindows()

# Calculate camera matrix based upon calibration results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

# Store calibration results in a zip file for future use
np.savez("cameraCalNums", ret, mtx, dist, rvecs, tvecs)