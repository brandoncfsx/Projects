import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

# Crop image: rows, cols.
cropped = image[30:60, 37:68]
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)