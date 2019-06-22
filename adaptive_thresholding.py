import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow('Image', image)
cv2.waitKey(0)

# Args: image, max value, method to compute threshold for current neighbor of pixels (we are computing the mean of the neighborhood of pixels as our threshold here), thresholding method (inverse: any pixel intensity greater than the threshold in the neighborhood is set to 0, otherwise it is set to 255), neighborhood size (must be odd integer, 11 x 11 pixel region), last arg is C: an integer that is subtracted from the mean, allowing us to fine-tune the threshold value.
# Mean weighted adaptive thresholding.
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow('Mean Thresholding', thresh)

# Gaussian (weighted mean) thresholding - indicated by the third arg.
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow('Gaussian Thresholing', thresh)
cv2.waitKey(0)