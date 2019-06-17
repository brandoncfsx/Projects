import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
(B, G, R) = cv2.split(image)

# Darker areas in a channel means there is not much of that color in that respective channel.
cv2.imshow('RED', R)
cv2.imshow('GREEN', G)
cv2.imshow('BLUE', B)
cv2.waitKey(0)

merged = cv2.merge([B, G, R])
cv2.imshow('Merged Image', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# An alternative way to visualize the channels is by setting the intensity of the channel pixels into its respective channel while setting the other channels to zero.
zeros = np.zeros(image.shape[:2], dtype='uint8')
cv2.imshow('RED', cv2.merge([zeros, zeros, R]))
cv2.imshow('GREEN', cv2.merge([zeros, G, zeros]))
cv2.imshow('BLUE', cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)