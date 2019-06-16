import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype='uint8')
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
# Create a rectangle with center at the center of the image.
cv2.rectangle(mask, (cX - 25, cY - 25), (cX + 25, cY + 25), 255, -1)
cv2.imshow('Mask', mask)
cv2.waitKey(0)

# By supplying the mask arg, the bitwise_and function only examines the pixels that are "on" in the mask.
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Mask Applied to Image', masked)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.circle(mask, (cX, cY), 25, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Mask Applied to Image', masked)
cv2.waitKey(0)