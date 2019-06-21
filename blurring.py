import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# Averaging Blurring.
# hstack: stacks images together horizontally. Allows us to use one window to display all three images.
# Blur args: image and size of kernel.
blurred = np.hstack([
    cv2.blur(image, (3, 3)),
    cv2.blur(image, (5, 5)),
    cv2.blur(image, (7, 7))])

cv2.imshow('Averaged', blurred)
cv2.waitKey(0)

# Gaussian Blurring.
# Args for GaussianBlur is the same for the averaging blurring except the third argument is our standard deviation in the x-axis direction. Setting this value to zero lets OpenCV automatically compute it based on our kernel size.
blurred = np.hstack([
    cv2.GaussianBlur(image, (3, 3), 0),
    cv2.GaussianBlur(image, (5, 5), 0),
    cv2.GaussianBlur(image, (7, 7), 0)])

cv2.imshow('Gaussian', blurred)
cv2.waitKey(0)

# Median Blurring.
blurred = np.hstack([
    cv2.medianBlur(image, 3),
    cv2.medianBlur(image, 5),
    cv2.medianBlur(image, 7)])

cv2.imshow('Median', blurred)
cv2.waitKey(0)

# Bilateral Blurring.
# Args: image, diameter of pixel neighborhood, color sigma (a larger value for color sigma means that more colors in the neighborhood will be considered when computing the blur), space sigma (a larger value for this means that pixels farther out from the central pixel will influence the blurring calculation, provided that their colors are similar enough).
blurred = np.hstack([
    cv2.bilateralFilter(image, 5, 21, 21),
    cv2.bilateralFilter(image, 7, 31, 31),
    cv2.bilateralFilter(image, 9, 41, 41)])

cv2.imshow('Bilateral', blurred)
cv2.waitKey(0)