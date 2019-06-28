from shapedetector import ShapeDetector
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
# Shrinking images allow for better approximation of shapes.
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / resized.shape[0]

# Convert to grayscale, blur, and apply threshold.
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Threshold value is manually set per image. Might also have to use THRESH_BINARY instead of the inverse depending on the input image.
thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]

# Find contours in thresholded image.
(cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sd = ShapeDetector()

for c in cnts:
    # Compute the center of the contour.
    M = cv2.moments(c)
    cX = int((M['m10'] / M['m00']) * ratio)
    cY = int((M['m01'] / M['m00']) * ratio)
    # Identify name of shape using contour.
    shape = sd.detect(c)

    # Multiply the contour (x, y) coordinates by resize ratio to get coorect coordinates for original image, then draw the contours and the name of the shape.
    c = c.astype('float')
    c *= ratio
    c = c.astype('int')
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)