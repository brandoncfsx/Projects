import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)
# Note: OpenCV stores RGB channels in reverse order.

(b, g, r) = image[0, 0]
print(f'Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}.')

# Change pixel color so we only have red. Remember BGR format.
image[0, 0] = (0, 0, 255)
(b, g, r) = image[0, 0]
print(f'Pixel at (0,0) - Red: {r}, Green: {g}, Blue: {b}.')

cv2.imshow('Corner', image[0:50, 0:50])
cv2.waitKey(0)

# Change top left corner of image to green. Indices are y coordinate (rows) and x coordinate (cols)
image[0:50, 0:50] = (0, 255, 0)

cv2.imshow('Updated', image)
cv2.waitKey(0)