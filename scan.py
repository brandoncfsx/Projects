from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# The threshold_local function helps us obtain the black and white feel to our scanned image.

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
# Compute ratio of old height to new height. We resize scanned image to have a height of 500 pixels to speedup image processing and also to make edge detection more accurate. The ratio will allow us to perform the scan on the original image rather than the resized image.
ratio = image.shape[0] / 500
orig = image.copy()
# Resize a copy of the image.
image = imutils.resize(image, height=500)

# Step 1: Edge Detection
# Convert to grayscale, blur, and find edges.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

print(f'STEP 1: Edge Detection')
cv2.imshow('Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()