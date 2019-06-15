import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

# When resizing an image, we need to keep in mind the aspect ratio (r) of the image in order to maintain correct ratio of the image.
r = 150.0 / image.shape[1]
# Set new image width to 150 and change height to ratio of new width divided by old width.
dim = (150, int(image.shape[0] * r))

# Args: image, computed dimensions of new image, and interpolation method - the algorithm that handles how the actual image is resized. Other choices for interpolation are INTER_LINEAR, INTER_CUBIC, INTER_NEAREST.
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Width)', resized)
cv2.waitKey(0)

r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Height)', resized)
cv2.waitKey(0)