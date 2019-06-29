import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute Scharr gradient magnitude representation of the images in both x and y direction. Scharr operator is specified using ksize=-1.
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Subtract y gradient from x gradient which gives us regions that have high horizontal gradients and low vertical gradients.
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow('Gradient', gradient)
cv2.waitKey(0)

# Next, filter out the noise and extrat ROI.
# Average blur/
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)

# In order to close the gaps in the barcode region in the thresholded image, we need to perform morphological operations. Create rectangular kernel with a width that is larger than the height which allows us to close the gaps between vertical stripes.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Closing', closed)
cv2.waitKey(0)

# Next, we need to remove random small blobs in the image that are not part of the barcode region by a series of erosions and dilations. Erosions remove the white blobs while dilation will grow the white regions outwards.
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
cv2.imshow('Erosion and Dilation', closed)
cv2.waitKey(0)