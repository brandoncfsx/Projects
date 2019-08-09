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
if h > 500 and w > 500:
    image = cv2.resize(image, (int(w * 0.8), int(h * 0.8)))
    (h, w) = image.shape[:2]
# Preprocess image - sets blob dimensions and normalization.
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the DNN and grab the detections.
print('COMPUTING OBJECT DETECTIONS ...')
net.setInput(blob)
detections = net.forward()

# Loop over the detections.
for i in range(0, detections.shape[2]):
    # Extract confidence associated with the detection.
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections.
    if confidence > args['confidence']:
        # Compute (x, y) coordinates of the bounding box.
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # Draw bounding box of the face along with its confidence.
        text = f'{(confidence * 100):.2f}'
        # Conditional to ensure text will not appear above the image.
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f'Detected {i} image(s).')

cv2.imshow('Output', image)
cv2.waitKey(0)