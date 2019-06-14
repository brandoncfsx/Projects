import numpy as np
import cv2

# Since pixels have value from 0 to 255, we can use an 8-bit unsigned integer.
canvas = np.zeros((300, 300, 3), dtype='uint8')

# Lines.
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

# Rectancles.
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

# Circles.
# Start off with a clean slate.
canvas = np.zeros((300, 300, 3), dtype='uint8')
# Set center of circle to middle of image. Shape of image is rows, cols, channels.
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
white = (255, 255, 255)

# Draw circles with increasing radius.
for radius in range(0, 175, 25):
    # Args: image, center of circle, radius, color.
    cv2.circle(canvas, (centerX, centerY), radius, white)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

# Abstract.
# Generate 25 circles.
for n in range(0, 25):
    # Radius value between 5 and 199.
    radius = np.random.randint(5, high=200)
    # List of three values between 0 and 255.
    color = np.random.randint(0, high=256, size=(3,)).tolist()
    # List of two values for center of circle between 0 and 299.
    pt = np.random.randint(0, high=300, size=(2,))

    cv2.circle(canvas, tuple(pt), radius, color, -1)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)

canvas = np.zeros((300, 300, 3), dtype='uint8')
# To draw every other square, skip two times the width.
increment = canvas.shape[0] // 30
# Since range's third arg is step size, divide the width of the image by the number of squares we want.
step_size = canvas.shape[0] // 15
for col in range(0, canvas.shape[1], step_size):
    for row in range(0, canvas.shape[0], step_size):
        cv2.rectangle(canvas, (col, row+increment), (col+increment-1, row+2*increment-1), red, -1)
        cv2.rectangle(canvas, (col+increment, row), (col+2*increment-1, row+increment-1), red, -1)
cv2.circle(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2), 50, green, -1)
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)