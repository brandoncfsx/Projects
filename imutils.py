import numpy as np
import cv2

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def rotate(image, degrees, scale=1.0, center=None):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, degrees, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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