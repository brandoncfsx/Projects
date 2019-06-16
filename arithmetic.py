import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

print(f'Max of 255: {cv2.add(np.uint8([200]), np.uint8([100]))}.')
print(f'Min of 0: {cv2.subtract(np.uint8([50]), np.uint8([100]))}.')

print(f'Wrap around: {np.uint8([200]) + np.uint8([100])}.')
print(f'Wrap around: {np.uint8([50]) - np.uint8([100])}')

# Image becomes brighter because we are increasing pixel values.
M = np.ones(image.shape, dtype='uint8') * 100
added = cv2.add(image, M)
cv2.imshow('Added via OpenCV', added)
cv2.waitKey(0)

# Image becomes darker because we are decreasing pixel values.
M = np.ones(image.shape, dtype='uint8') * 50
subtracted = cv2.subtract(image, M)
cv2.imshow('Subtracted via OpenCV', subtracted)
cv2.waitKey(0)