import numpy as np
import cv2

def translate(image, x, y):
    """
    Translate image horizontally and/or vertically.
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, degrees, scale=1.0, center=None):
    """
    Rotate and scale an image.

    Keyword arguments:
    center -- location from which we will rotate the image.
    """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, degrees, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize the image while maintaining aspect ratio.

    Keyword arguments:
    inter -- method of interpolation.
    """
    if height == None:
        r = width / image.shape[1]
        dim = (width, int(image.shape[0] * r))

    elif width == None:
        r = height / image.shape[0]
        dim = (int(image.shape[1] * r), height)

    else:
        print(f'Please enter either width or height.\nReturning default image.')
        return image

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts