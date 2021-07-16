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
x1 = 570
y1 = 437
x2 = 196
y2 = 687
x3 = 1076
y3 = 648
x4 = 760
y4 = 430

#3 - Hough transformation variables
rho = 3
theta = 8 * np.pi/180
threshold = 20
minLength = 20
maxGap = 5

# Importing the test video
cap = cv2.VideoCapture("test_videos/challenge.mp4")
if not cap.isOpened():
    raise BrokenPipeError("Video not initializing")

# Defining codecs and videoWriter for creating output video
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
output = cv2.VideoWriter("test_videos_output/challenge_final.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()

    if not ret: # If frame is not read, video has ended, exit
        break

    # Splitting the image into three channels and then removing green - helps detect yellow better
    b, g, r = cv2.split(frame)
    img = cv2.merge((b, np.zeros_like(b), r))

    # Blurring the image in hopes it improves detection of lane line
    img = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)

    # Canny edge detection run on image
    canny = cv2.Canny(img, 41, 106)

    # Creating polygon mask
    mask = np.zeros_like(canny)
    imshape = img.shape
    vertices = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)

    # Applying mask and hough transform
    masked_edges = cv2.bitwise_and(canny, mask)
    linesP = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLength, maxGap)

    # Drawing hough lines onto original frame for output
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 255, 100), 2, cv2.LINE_AA)

    # Writing the modified frame to the output video
    output.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(19) == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
