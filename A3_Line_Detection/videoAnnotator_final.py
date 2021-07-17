import numpy as np
import cv2
import numpy.polynomial.polynomial as npp

"""
Variable setup-
1 - Canny thresholds
2 - Mask polygon co-ords
3 - Hough transformation vars
"""
# 1 - Lower/Upper Canny thresholds
# Depreciated - Variables have to manually be set in the canny function otherwise the code breaks.

#2 - Points 1-4 co-ordinates for mask
x1 = 422
y1 = 332
x2 = 123
y2 = 540
x3 = 897
y3 = 540
x4 = 551
y4 = 323

#3 - Hough transformation variables
rho = 2
theta = 1 * np.pi/180
threshold = 23
minLength = 3
maxGap = 21

"""
Video import/export setup
"""
# Importing the test video
cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
if not cap.isOpened():
    raise BrokenPipeError("Video not initializing")

# Defining codecs and videoWriter for creating output video
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
output = cv2.VideoWriter("annotated_videos/solidYellowLeft_final.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

"""
Video processing start
"""
while True:
    ret, frame = cap.read()

    if not ret: # If frame is not read, video has ended, exit
        break

    # Image converted to grayscale for processing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny edge detection run on image
    canny = cv2.Canny(img, 174, 173)

    # Creating polygon mask
    mask = np.zeros_like(canny)
    imshape = img.shape
    vertices = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    # Applying mask and hough transform
    masked_edges = cv2.bitwise_and(canny, mask)
    linesP = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLength, maxGap)

    # Drawing lane lines based upon average of hough lines
    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []
    if linesP is not None: # Determining whether a line is horizontal (discarded) or to the left or right of the image
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # l0 = x1, l1 = y1, l2 = x2, l3 = y2
            # cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 255, 100), 2, cv2.LINE_AA)
            if (l[3]-l[1]) / (l[2]-l[0]) > 0.25 or (l[3]-l[1]) / (l[2]-l[0]) < -0.25:
                if l[0] <= (int(cap.get(3)) //2 ):
                    left_points_x.append(l[0])
                    left_points_y.append(l[1])
                    left_points_x.append(l[2])
                    left_points_y.append(l[3])
                else:
                    right_points_x.append(l[0])
                    right_points_y.append(l[1])
                    right_points_x.append(l[2])
                    right_points_y.append(l[3])

    # Using linear regression to find linear equations representing left and right lines
    left_line = None
    right_line = None
    if left_points_x and left_points_y:
        left_line = npp.polyfit(left_points_y, left_points_x, 1)
    if right_points_x and right_points_y:
        right_line = npp.polyfit(right_points_y, right_points_x, 1)

    # Drawing the left and right lines using co-ords determined from line equations
    if left_line is not None:
        cv2.line(frame, (int(npp.polyval(y1, left_line)), y1),
                 (int(npp.polyval(y2, left_line)), y2), (0, 255, 100), 4, cv2.LINE_AA)
    if right_line is not None:
        cv2.line(frame, (int(npp.polyval(y3, right_line)), y3),
                 (int(npp.polyval(y4, right_line)), y4), (0, 255, 100), 4, cv2.LINE_AA)

    # Writing the modified frame to the output video
    output.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(19) == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
