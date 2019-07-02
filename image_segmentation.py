import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb

flags = [i for i in dir(cv2) if i.startswith("COLOR_")]
# There are 258 flags. The first characters after COLOR_ indicate the origin color space and the characters after the '2' is the target color space.
print(len(flags), "flags total:")
print(flags[40])

# Reads image in BGR space.
nemo = cv2.imread("nemo5.jpg")
cv2.imshow('BGR', nemo)

nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
cv2.imshow('RGB', nemo)
cv2.waitKey(0)

# Plotting the image on 3D plot, each axis represents a channel of the color space.
r, g, b = cv2.split(nemo)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
# Flatten every pixel into a list and normalize it to pass into the facecolors parameter of matplotlib scatter function. Normalizing means to condense the range from 0-255 to 0-1. The facecolors parameter also wants a list and not a NumPy array.
pixel_colors = nemo.reshape((np.shape(nemo)[0] * np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1.0, vmax=1.0)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# Build scatter plot.
axis.scatter(
    r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker="."
)
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
# Can see that the orange parts of the image span across almost the entire range of red, green and blue values. So segmenting by ranges of RGB values will not be easy.
plt.show()

# HSV (hue, saturation, and value(brightness)) is a good choice of color space for segmenting by color. The colors are modeled as an angular dimension rotating around a central vertical axis which represents the value channel. Value ranges from dark (0 at bottom) to light at top. Third axis, saturation defines the shades of hue from least saturated at the vertical axis to most saturated furthest away from the center.
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
# We can see that the oranges are much more localized and visually separable. There is very little variation along the hue axis, which is the key point that can be leveraged for segmentation.

# Pick out a range by eyeballing the plot. Parameters are H, S, and V.
light_orange = (1, 190, 100)
dark_orange = (18, 255, 255)
# Normalise to 0 - 1 range for viewing with matplotlib. Here, we will display a 10x10x3 NumPy array of both the light and dark orange colors.
lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0

# Remember colors are in HSV space, so need to convert to RGB.
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()

# Threshold nemo image, this will return a binary mask with the size of the image where values of 1 indicate values within the range.
mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result = cv2.bitwise_and(nemo, nemo, mask=mask)

# Display the mask and its application on the original image.
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()

# Next, let's also segment the white stripes.
light_white = (0, 0, 200)
dark_white = (145, 60, 255)
lw_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
dw_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(dw_square))
plt.show()
# Can see that the upper range of the white stripes has some blue.

mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

# Combine masks for orange and white colors by addition operation.
final_mask = mask + mask_white
final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)

plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()

# Apply Guassian blur to tidy up the small false detections such as a few stray pixels along the segmentation border.
blur = cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()

friend = cv2.cvtColor(cv2.imread('nemo5.jpg'), cv2.COLOR_BGR2RGB)

def segment_fish(image):
    '''
    Algorithm to perform image segmentation on an orange clownfish.
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)
    mask = cv2.inRange(hsv_image, light_orange, dark_orange)
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)
    final_mask = mask + mask_white
    result = cv2.bitwise_and(image, image, mask=final_mask)
    result = cv2.GaussianBlur(result, (7, 7), 0)
    return result

result = segment_fish(friend)

plt.subplot(1, 2, 1)
plt.imshow(friend)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()