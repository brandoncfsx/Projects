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
# Threshold value is manually set per image.
thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)[1]

# Find contours in thresholded image.
(cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sd = ShapeDetector()
