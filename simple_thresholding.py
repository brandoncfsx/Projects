import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
ap.add_argument('-t', '--threshold', type=int, default=128, help='Threshold value.')
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

# We reveal the objects in the image and hide everything else by masking the image with the inverted threshold image. This is because a mask only considers pixels in the original image where the mask is greater than zero. Since the inverted thresholded image can approximate the areas the objects are contained, we can use this inverted thresholded image as our mask.
cv2.imshow('Coins', cv2.bitwise_and(image, image, mask=thresholdInv))
cv2.waitKey(0)

# THRESH_TRUNC: leaves pixel intensities as they are if the source pixel is not greater than the supplied threshold.

# THRESH_TOZERO: sets the source pixel to zero if the source pixel is not greater than the supplied threshold.

otherMethods = [
    ('THRESH,TRUNC', cv2.THRESH_TRUNC),
    ('THRESH_TOZERO', cv2.THRESH_TOZERO),
    ('THRESH_TOZERO_INV', cv2.THRESH_TOZERO_INV)]

for (threshName, threshMethod) in otherMethods:
    (T, threshold) = cv2.threshold(image, args['threshold'], 255, threshMethod)
    cv2.imshow(threshName, threshold)
    cv2.waitKey(0)

# Note: these methods require human-intervention: fine-tuing of the threshold value.