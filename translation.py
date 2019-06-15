import numpy as np
import argparse
import cv2
# imutils is a library we are going to write and create helper methods to do common tasks like translation, rotation, and resizing.
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

# Define a translation matrix M which is a floating point array.
# First row is [1,0,tx], tx is the number of pixels we will shift horizontally. Negative values shift left and positive values shift to the right.
# Second row is [0,1,ty], ty shifts vertically, negative shifts up and positive values shift the image down.
# So the code below: shift to the right and downwards.
M = np.float32([[1, 0, 25], [0, 1, 50]])
# Translation occurs here - args: image, translation matrix, and dimensions of our image (x, y).
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted down and right', shifted)
cv2.waitKey(0)

# Can see that it takes quite a bit of code to translate an image, instead we will create a helper method in imutils.py.
M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted up and left', shifted)
cv2.waitKey(0)