from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True, help='Path to Caffe "deploy" prototxt file.')
ap.add_argument('-m', '--model', required=True, help='Path to Caffe model.')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections.')
args = vars(ap.parse_args())

# Load model.
print('LOADING MODEL ...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Initialize video stream.
print('STARTING VIDEO STREAM ...')
# VideoStream object will target first camera detected.
vs = VideoStream(src=0).start()
# For Raspberry Pi: replace src=0 with usePiCamera=True.
# For video file: VideoStream class to FileVideoStream.

while True:
    # Grab frame and resize it.
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Grab frame dimensions and convert it to a blob.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass blob into network and grab detections.
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args['confidence']:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        text = f'{(confidence * 100):.2f}'
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.imshow ('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # If press 'q' key, break from loop.
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()