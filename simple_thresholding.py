import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply a Gaussian blurring with sigma equal to radius of five to grayscale image. Applying Gaussian blurring helps remove some of the high frequency edges in the image.
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('Image', image)
cv2.waitKey(0)

# Args: grayscale image, threshold value, maximum value (any pixel intensity greater than our threshold is set to this value). So any pixel value greater than 155 is set to 255. Last arg is the thresholding method. Return values are our threshold value and the thresholded image.
(T, threshold) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', threshold)

(T, thresholdInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Inverse Threshold Binary', thresholdInv)

cv2.imshow('Coins', cv2.bitwise_and(image, image, mask=thresholdInv))
cv2.waitKey(0)