import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    print(x)

src = "test_images/challenge_2.png"
img = cv2.imread(src, cv2.IMREAD_COLOR) #read image as grayscale

b,g,r = cv2.split(img)
img = cv2.merge((b, np.zeros_like(b), r))
canny = cv2.Canny(img, 85, 255) # Run canny edge detection on image

mask = np.zeros_like(canny) # Creating a polygon mask
imshape = img.shape
vertices = np.array([[(0,0), (0,imshape[0]), (imshape[1],imshape[0]), (imshape[1],0)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, 255)
masked_edges = cv2.bitwise_and(canny, mask)
cp_masked_edges = np.copy(masked_edges)
linesP = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 15, np.array([]), 40, 20) # Run hough transformation on mask
lines_out = np.copy(img)

if linesP is not None: # Drawing hough lines
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(lines_out, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.namedWindow('image')  # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback)  # lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback)  # upper threshold trackbar for window 'image

cv2.namedWindow("poly")  # window for perfecting polymask
cv2.createTrackbar("X1", 'poly', 0, imshape[1], callback)
cv2.createTrackbar("Y1", 'poly', 0, imshape[0], callback)
cv2.createTrackbar("X2", 'poly', 0, imshape[1], callback)
cv2.createTrackbar("Y2", 'poly', imshape[0], imshape[0], callback)
cv2.createTrackbar("X3", 'poly', imshape[1], imshape[1], callback)
cv2.createTrackbar("Y3", 'poly', imshape[0], imshape[0], callback)
cv2.createTrackbar("X4", 'poly', imshape[1], imshape[1], callback)
cv2.createTrackbar("Y4", 'poly', 0, imshape[0], callback)

cv2.namedWindow("output")  # Output window to show lines
cv2.createTrackbar("Rho", 'output', 1, 10, callback)  # Trackbars for hough transformation params
cv2.createTrackbar("Theta", 'output', 1, 360, callback)
cv2.createTrackbar("Threshold", 'output', 15, 100, callback)
cv2.createTrackbar("minLength", 'output', 40, 100, callback)
cv2.createTrackbar("maxGap", 'output', 20, 100, callback)


while(1):
    cv2.imshow('image', canny)
    cv2.imshow('poly', cp_masked_edges)
    cv2.imshow('output', lines_out)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        with open("test_images_output/Challenge_output.txt", 'w') as file: #Writing all trackbar data to output
            file.write(src)
            file.write("\n")
            file.write(str(l))
            file.write("\n")
            file.write(str(u))
            file.write("\n")
            file.write(str(rho))
            file.write("\n")
            file.write(str(theta))
            file.write("\n")
            file.write(str(threshold))
            file.write("\n")
            file.write(str(minL))
            file.write("\n")
            file.write(str(maxG))
            file.write("\n")
            file.write(f"({x1//2}, {y1//2}), ")
            file.write(f"({x2//2}, {y2//2}), ")
            file.write(f"({x3//2}, {y3//2}), ")
            file.write(f"({x4//2}, {y4//2})")
        break
    # Loading all the data from the trackbars
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')
    x1 = cv2.getTrackbarPos('X1', 'poly')
    x2 = cv2.getTrackbarPos('X2', 'poly')
    x3 = cv2.getTrackbarPos('X3', 'poly')
    x4 = cv2.getTrackbarPos('X4', 'poly')
    y1 = cv2.getTrackbarPos('Y1', 'poly')
    y2 = cv2.getTrackbarPos('Y2', 'poly')
    y3 = cv2.getTrackbarPos('Y3', 'poly')
    y4 = cv2.getTrackbarPos('Y4', 'poly')
    rho = cv2.getTrackbarPos("Rho", 'output')
    theta = cv2.getTrackbarPos("Theta", 'output')
    threshold = cv2.getTrackbarPos("Threshold", 'output')
    minL = cv2.getTrackbarPos("minLength", 'output')
    maxG = cv2.getTrackbarPos("maxGap", 'output')

    canny = cv2.Canny(img, l, u)

    mask = np.zeros_like(canny)
    vertices = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(canny, mask)
    cp_masked_edges = np.copy(masked_edges)
    cv2.circle(cp_masked_edges, (x1, y1), 5, 255, -1)
    cv2.circle(cp_masked_edges, (x2, y2), 5, 255, -1)
    cv2.circle(cp_masked_edges, (x3, y3), 5, 255, -1)
    cv2.circle(cp_masked_edges, (x4, y4), 5, 255, -1)

    if rho < 1:
        rho = 1
    if theta < 1:
        theta = 1

    linesP = cv2.HoughLinesP(masked_edges, rho, theta * np.pi / 180, threshold, np.array([]), minL, maxG)
    lines_out = np.copy(img)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(lines_out, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.destroyAllWindows()