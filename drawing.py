import numpy as np
import cv2

# Since pixels have value from 0 to 255, we can use an 8-bit unsigned integer.
canvas = np.zeros((300, 300, 3), dtype='uint8')

# Let's draw lines.
green = (0, 255, 0)
# Arguments for line method is image, start point, end point and color.
cv2.line(canvas, (0, 0), (300, 300), green)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

red = (0, 0, 255)
# Last argument is the thickness of the line in pixels.
cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Let's draw rectancles.
# Second arg: start position, third: end position.
# Region of rectangle is (end1-start1)x(end2-start2) pixels.
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Remember, position arguments are of form pixels in x direction and then pixels in y direction, and are not representative of rows x cols.
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

blue = (255, 0, 0)
# A negative value for thickness fills the rectangle instead of defining thinkness of the perimeter of the rectangle.
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)