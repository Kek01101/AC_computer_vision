import numpy as np
import cv2


"""
Variable setup-
1 - Canny thresholds
2 - Mask polygon co-ords
3 - Hough transformation vars
"""
# 1 - Lower/Upper Canny thresholds
# Depreciated - Variables have to manually be set in the canny function otherwise the code breaks.

#2 - Points 1-4 co-ordinates for mask
x1 = 277 * 2.02
y1 = 351 * 2.12
x2 = 0
y2 = 535 * 2.12
x3 = 680 * 2.02
y3 = 535 * 2.12
x4 = 454 * 2.02
y4 = 306 * 2.12

#3 - Hough transformation variables
rho = 2
theta = 2 * np.pi/180
threshold = 15
minLength = 15
maxGap = 15

# Importing the test video
cap = cv2.VideoCapture("test_videos/Example002.mp4")
if not cap.isOpened():
    raise BrokenPipeError("Video not initializing")

# Defining codecs and videoWriter for creating output video
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
output = cv2.VideoWriter("test_videos_output/2_final.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()

    if not ret: # If frame is not read, video has ended, exit
        break

    # Image converted to grayscale for processing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny edge detection run on image
    canny = cv2.Canny(img, 173, 106)

    # Creating polygon mask
    mask = np.zeros_like(canny)
    imshape = img.shape
    vertices = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    # Applying mask and hough transform
    masked_edges = cv2.bitwise_and(canny, mask)
    linesP = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLength, maxGap)

    # Drawing hough lines onto original frame
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)

    # Writing the modified frame to the output video
    output.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(19) == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
