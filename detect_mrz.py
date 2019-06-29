import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

# Initialize a rectangular and square structuring kernel for morphological operations. The rectangle kernel's width is about 3x larger than its height.
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

image = cv2.imread(args['image'])
image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Smooth via 3x3 Gaussian (removes high frequency noise), then apply a blackhat morphological operator to find dark regions on a light background.
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# Reveals black text against the light passport background.
cv2.imshow('Blackhat', blackhat)
cv2.waitKey(0)

# Next step in MRZ detection is to compute the gradient magnitude representation of the blackhat image using the Scharr operator. We find the Scharr gradient along the x-axis which reveals regions that are not only dark against a light background, but also contain vertical changes in the gradient, such as the MRZ text region.
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# Scale into range [0, 255] using min/max scaling.
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype('uint8')
cv2.imshow('Gradient', gradX)
# This step is extremely helpful in reducing false-positive MRZ detections. Without it, we can accidentally mark embellished or designed regions of the passport as MRZ.

# Apply a closing operation using the rectangular kernel to close gaps in between characters, then apply Otsu's thresholding method.
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Horizontal Closing', gradX)
cv2.imshow('Thresholded', thresh)

# Close gap between lines to get a rectangular region by using the square kernel. Then apply a series of erosions to break apart connected components. Erosions help to remove small blobs that are irrelevant to the MRZ.
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=4)
cv2.imshow('Vertical Closing', thresh)

# Set 5% of the left and right borders of the image to zero (black) since the border of the passport may have become attached to the MRZ region during closing operations.
p = int(image.shape[1] * 0.05)
thresh[:, 0:p] = 0
thresh[:, image.shape[1] - p:] = 0
cv2.imshow('Border Removal', thresh)
cv2.waitKey(0)

# Lastly, find the contours and use their properties to find the MRZ.
(cnts, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Sort contours based on size in descending order.
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
roi = image.copy()
print(f'Number of contours: {len(cnts)}')

for c in cnts:
    # Compute bounding box of the contour and use it to find the aspect ratio and coverage ratio of the bounding box width to the width of the image.
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / h
    crWidth = w / gray.shape[1]
    print(f'Aspect ratio: {ar}, Coverage Ratio: {crWidth}')

    # Check to see if the aspect ratio and coverage width are acceptable. The MRZ is rectangular, with a width that is much larger than the height, and should spanm at least 70% of the input image.
    if ar > 5 and crWidth > 0.7:
        # Pad the bounding box since we applied erosions.
        pX = int((x+w) * 0.03)
        pY = int((y+h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX*2), h + (pY*2))

        # Extract the ROI and draw a bounding box around the MRZ.
        roi = image[y : y+h, x : x+w].copy()
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

cv2.imshow('Image', image)
cv2.imshow('ROI', roi)
cv2.waitKey(0)