from __future__ import print_function
# Argparse: parse command line arguments.
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Run script via: pipenv run python load_display_save.py --image ../../../Trio.png
# For additional arguments, just repeat such as --image hi.png --image_2 hi2.png.

# Parse command line arguments. Arg --image is the path to our image.
ap = argparse.ArgumentParser()
# Can add multiple arguments via ap.add_argument(...).
ap.add_argument('-i', '--image', required=True, help='Path to the image')
# Store arguments in a dictionary.
args = vars(ap.parse_args())

# Loading.
# Imread function returns a NumPy array representing the image.
# Shape of NumPy array is (height(rows), width(cols), channel(s)).
# Argument of args must be equal to second arg in add_argument method without '--'.
image = cv2.imread(args['image'])
print(f'Width: {image.shape[1]} pixels.')
print(f'Height: {image.shape[0]} pixels.')
print(f'Channels: {image.shape[2]}.')

# Displaying.
# First arg is name of our window. Second arg is reference to image.
cv2.imshow('Image', image)
# waitKey to pause execution of script until we press a key.
cv2.waitKey(0)
# Or we can use matplotlib.
# Need to reverse BGR to RGB for matplotlib. Note: images stored in opencv2 is in BGR.
# Note: imshow method for matplotlib does not work for displaying frames of a video stream or video file.
plt.imshow(image[:,:,::-1])
plt.axis('off')
plt.show()

# Saving.
# Write image to file in JPG format.
# First arg: path to file, second is image we want to save.
# This function just lets us save the image in another format like .jpg.
cv2.imwrite('newimage.jpg', image)