import numpy as np
import argparse
import cv2

def auto_canny(image, sigma=0.33):
    '''
    Automatically determine upper and lower boundaries for Canny edge detection.

    Key arguments:
    sigma - varies the percentage thresholds that are determined based on statistics. A lower value indicates a tighter threshold, and a larger value gives a wider threshold. Good practice to start with default of 33%.
    '''

    # Compute the median of the single channel pixel intensities.
    v = np.median(image)

    # Apply automatica Canny edge detection using the median.
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Canny edge detection using three methods: wide boundary, tight boundary, and automatic.
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

cv2.imshow('Original', image)
cv2.imshow('Edges', np.hstack([wide, tight, auto]))
cv2.waitKey(0)