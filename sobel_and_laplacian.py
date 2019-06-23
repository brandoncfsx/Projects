import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)

# Use Laplacian method to compute the gradient magnitude image. The second arg is our data type for the output image.
lap = cv2.Laplacian(image, cv2.CV_64F)
# Take absolute value of the gradient image and convert back to unsigned 8-bit integer.
lap = np.uint8(np.absolute(lap))
cv2.imshow('Laplacian', lap)
cv2.waitKey(0)

# Compute the Sobel gradient representation.
# Last two args are the order of the derivatives in the x and y direction, respectively. Values of 1 and 0 to find vertical edge-like regions, and 0 and 1 to find horizontal edge-like regions.
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

# These lines ensure we find all edges.
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combine gradient images in both the x and y direction.
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow('Sobel X', sobelX)
cv2.imshow('Sobel Y', sobelY)
cv2.imshow('Sobel Combined', sobelCombined)
cv2.waitKey(0)