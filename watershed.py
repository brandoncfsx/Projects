from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
# Pyramid mean shift filter to help with thresholding.
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow('Image', image)

# Convert the shifted image to grayscale then apply Otsu's thresholding.
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)

# First, compute the exact Euclidean distance transform from every binary pixel to the nearest zero pixel, then find peaks in this distance map.
# D is our distance map.
D = ndimage.distance_transform_edt(thresh)
# Find peaks (local maxima) in our distance map, ensuring that there is at least a 20 pixel distance between each peak.
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

# Take the output of the peak_local_max function to compute a connected component analysis using 8-connectivity.
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
# Input our markers, and since watershed assumes our markers represent local minima in our distance map, we take the negative value of D.
labels = watershed(-D, markers, mask=thresh)
# Labels is a NumPy array with the same dimensions as our image. Each pixel value has a unique label value so that pixels that have the same label value belong to the same object.
print(f'{len(np.unique(labels)) - 1} unique segments found')

# Loop over the unique label values and extract each of the unique objects.
for label in np.unique(labels):
    # If label is zero, it is the background, so ignore it.
    if label == 0:
        continue

    # Otherwise, allocate memory for the label region and draw it on the mask. Set pixels that belong to the current label to 255.
    mask = np.zeros(gray.shape, dtype='uint8')
    mask[labels == label] = 255

    # Detect contours in the mask and grab the largest one which will represent the outline of a given object in the image.
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Draw a circle enclosing the object.
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, f'#{label}', (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imshow('Output Image', image)
cv2.waitKey(0)