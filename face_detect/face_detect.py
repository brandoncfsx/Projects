# Use pre-trained deep neural network.
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image.')
ap.add_argument('-p', '--prototxt', required=True, help='Path to Caffe "deploy" prototxt file.')
ap.add_argument('-m', '--model', required=True, help='Path of Caffe pre-trained model.')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections.')
args = vars(ap.parse_args())

# Load model as 'net' and create a blob from the image.
print('LOADING MODEL ...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Load input image and create an input blob for it by resizing to a fixed size and then normalize it.
image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
# Preprocess image - sets blob dimensions and normalization.
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))