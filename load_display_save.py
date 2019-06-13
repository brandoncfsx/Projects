from __future__ import print_function
# Argparse: parse command line arguments.
import argparse
import cv2

# Parse command line arguments. Arg --image is the path to our image.
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
# Store arguments in a dictionary.
args = vars(ap.parse_args())

# Imread function returns a NumPy array representing the image.
image = cv2.imread(args['image'])
print(f'Width: {image.shape[1]} pixels.')
print(f'Height: {image.shape[0]} pixels.')
print(f'Channels: {image.shape[2]}.')

# First arg is name of our window. Second arg is reference to image.
cv2.imshow('Image', image)
# waitKey to pause execution of script until we press a key.
cv2.waitKey(0)

# Write image to file in JPG format.
# First arg: path to file, second is image we want to save.
cv2.imwrite('newimage.jpg', image)