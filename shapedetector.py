import cv2

class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        # Initialize shape's name and approximate contour based on the contour input.
        shape = 'Unknown'
        # First compute the perimeter of the contour.
        peri = cv2.arcLength(c, True)
        # Contour approximation reduces the number of points in a curve with a reduced set of points: split-and-merge algorithm. Common values for the second arg is between 1 and 5% of the original contour perimeter.
        approx = cv2.approxPolyDP(c, 0.04*peri, True)

        # A contour consists of a list of vertices so we can check the number of entries in the list to determine the shape.
        if len(approx) == 3:
            shape = 'Triangle'

        elif len(approx) == 4:
            # Compute bounding box of the contour and use it to compute the aspect ratio in order to determine if this is a square or a rectangle.
            (x, y, w, h) = cv2.boundingRect(approx)
            # Our aspect ratio.
            ar = w / h

            shape = 'square' if ar >= 0.95 and ar <= 1.05 else 'rectangle'

        elif len(approx) == 5:
            shape = 'Pentagon'

        else:
            shape = 'Circle'

        return shape