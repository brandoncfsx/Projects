from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

chans = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Flattened Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

# For each channel we compute a histogram and setting the color attribute of the plot to the respective channel.
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)

fig = plt.figure()

ax = fig.add_subplot(131)
# First channel in channel list arg is on the y-axis.
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and B')
plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and R')
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and R')
plt.colorbar(p)

print(f'2D Histogram Shape: {hist.shape}, with {hist.flatten().shape[0]} values.')

plt.show()

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print(f'3D Histogram Shape: {hist.shape}, with {hist.flatten().shape[0]} values.')
