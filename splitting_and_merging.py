import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
(B, G, R) = cv2.split(image)

cv2.imshow('RED', R)
cv2.moveWindow('RED', 20, 20)
cv2.imshow('GREEN', G)
cv2.imshow('BLUE', B)
cv2.waitKey(0)

merged = cv2.merge([B, G, R])
cv2.imshow('Merged Image', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()