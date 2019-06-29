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

# Step 2: Finding Contours
# We will assume that the largest contour with exactly four points is our piece of paper to be scanned.
(cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    # If approximated contour has four points, then we found it.
    if len(approx) == 4:
        screenCnt = approx
        break

print(f'STEP 2: Find contours of paper')
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Apply a Perspective Transform and Threshold
# Apply four point transformation. Second arg is the contour of the document multiplied by the resize ratio. We do this because we performed edge detection and found contours on the resized image of height of 500 pixels.
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Produce black and white paper effect via grayscale conversion and thresholding.
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# Adaptive thresholding.
warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)
warped = warped.astype('uint8')

print(f'STEP 3: Apply Perspective Transform')
cv2.imshow('Original', imutils.resize(orig, height=650))
cv2.imshow('Scanned', imutils.resize(warped, height=650))
cv2.waitKey(0)