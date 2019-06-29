import numpy as np
import cv2

def order_points(pts):
    '''
    Sorts a list of four points into: top-left, top-right, bottom-right, bottom-left.

    Key arguments:
    pts -- A list of four (x, y) coordinate points of each point of a rectangle.
    '''


    # Initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left. Second entry is top-right, third is bottom-right and fourth is bottom-left.
    rect = np.zeros((4, 2), dtype='float32')

    # Top-left point will have the smallest x + y sum, the bottom-right point has the largest x + y sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points, the top-right point will have the smallest difference and the bottom-left has the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    '''
    

    Key arguments:
    image -- image we want to apply the perspective transform to.
    pts -- list of four points that contain the ROI of the image we want to transform.
    '''

    # Obtain a consistent order of the points and unpack them.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Determine the dimensions of our new warped image. New width and height is the max between the two pairs of options we have shown below.

    # Compute the width of the new image, which is the max distance between bottom-right and bottom-left x coordinates or the top-right and top-left x coordinates.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image which is the max distance between the top-right and bottom-right y coordinates or the top-left and bottom-left y coordinates.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # We now have the dimensions of our new image. Construct the set of points to get a bird's eye view of the image in ordered list tl-tr-br-bl.
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')

    # Compute the perspective transform matrix and apply it to get the top-down view. Args: list of the four ROI points in the original image, second is the list of transformed points.
    M = cv2.getPerspectiveTransform(rect, dst)
    # Args: image, transformation matrix, and width and height of output image.
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped