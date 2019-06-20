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