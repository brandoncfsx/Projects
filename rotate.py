import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

# When rotating an image, we need to specify around which point we want to rotate. In most cases, we will want to rotate around the center of an image.
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Define a rotation matrix. Instead of creating a matrix with NumPy, we use the getRotationMatrix2D method from OpenCV.
# Args: point at which we want to rotate the image around, then the number of degrees we are going to rotate the image by, and the scale of the image (1.0 keeps dimensions, 0.5 halves it, etc).
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow('Rotated by 45 degrees', rotated)
cv2.waitKey(0)

rotated = imutils.rotate(image, -90)
cv2.imshow('Rotated by -90 degrees', rotated)
cv2.waitKey(0)