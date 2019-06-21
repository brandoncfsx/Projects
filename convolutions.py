from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, kernel):
    '''
    Convolve an image with a kernel.

    Key arguments:
    image -- grayscale image.
    Kernel -- NumPy array with odd integer dimensions.
    '''
    # Get dimensions of image and kernel.
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Pad the borders of the image so the dimensions of the output image is not reduced.
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype='float32')

    # Loop over the input image and slide the kernel across each (x, y) coordinate one pixel at a time.
    for y in np.arange(iH):
        for x in np.arange(iW):
            # ROI is centered around the current (x, y) coordinate of the image. Dimension of ROI is the same as the kernel.
            roi = image[y : y+2*pad+1, x : x+2*pad +1]

            # Convolution on ROI and the kernel, then sum the matrix.
            k = (roi * kernel).sum()

            # Store output value into the output image (x, y) coordinate.
            output[y, x] = k

    # Rescale pixel intensity of output image to [0, 255] since convolutions can return values outside that range.
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')

    return output

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

# Create average blurring kernels to smooth the image.
smallBlur = np.ones((7, 7), dtype='float') * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype='float') * (1.0 / (21 * 21))

# Create a sharpening filter. This enhances line structures and other details.
sharpen = np.array((
    [0, 1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype='int')

# Laplacian kernel for edge detection.
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype='int')

# Sobel x-axis kernel. This detects vertical changes in the gradient of the image.
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype='int')

# Sobel y-axis kernel. This detects horizontal changes in the gradient of the image.
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype='int')

# Create a set of tuples called a kernel bank.
kernelBank = (
    ('small_blur', smallBlur),
    ('large_blur', largeBlur),
    ('sharpen', sharpen),
    ('laplacian', laplacian),
    ('sobel_x', sobelX),
    ('sobel_y', sobelY)
)

# Apply kernelBank to the image.
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Loop over the kernels.
for (kernelName, kernel) in kernelBank:
    # Apply kernel to the image using our convolve function and OpenCV's filter2D function.
    print(f'Applying {kernelName} kernel.')
    convolveOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    cv2.imshow('Original', gray)
    cv2.imshow(f'{kernelName} - convolve', convolveOutput)
    cv2.imshow(f'{kernelName} - OpenCV', opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()