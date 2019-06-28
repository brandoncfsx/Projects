import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
# Works with .mp4, .gif, etc.
ap.add_argument('-v', '--video', help='Path to the video file.')
args = vars(ap.parse_args())

# Load video.
camera = cv2.VideoCapture(args['video'])

while True:
    # Grab current frame and initialize status text. Return vals: a Boolean indicating if the frame was read successfully and the frame.
    (grabbed, frame) = camera.read()
    status = 'No targets.'

    # End if video is stopped.
    if not grabbed:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours in edge map.
    (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # Approximate the contours by using the split-and-merge algorithm which represents the contour as a series of short line segments.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # Ensure that the approximated contour is roughly rectangular. Note: Ideally a square should be represented by exactly four line segments, but margin is added due to image quality and/or noise.
        if len(approx) >= 4 and len(approx) <= 6:
            # Compute bounding box of the approximated contour and use this box to compute the aspect ratio.
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / h

            # Compute the Solidity of the original contour.
            # Two contour properties. Area of the bounding box = number of non-zero pixels inside the bounding box divided by  the total number of pixels in it.
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            # Solidity quantifies the amount and size of concavities in an object boundary.
            solidity = area / hullArea

            # Compute if the width and height, solidity and aspect ratio of the contour falls within appropriate bounds.
            keepDims = w>25 and h>25
            keepSolidity = solidity > 0.9
            # The range of values for aspect ratio is approximately a square.
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

            # Ensure that the contour passes all tests.
            if keepDims and keepSolidity and keepAspectRatio:
                # Draw an outline around the target.
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 4)
                status = 'Target(s) Found.'

                # Compute the center of the contour region and draw a crosshair.
                M = cv2.moments(approx)
                (cX, cY) = (int(M['m10'] // M['m00']), int(M['m01'] // M['m00']))
                (startX, endX) = (int(cX - (w*0.15)), int(cX + (w*0.15)))
                (startY, endY) = (int(cY - (h*0.15)), int(cY + (h*0.15)))
                cv2.line(frame, (startX, cY), (endX, cY), (0, 255, 0), 3)
                cv2.line(frame, (cX, startY), (cX, endY), (0, 255, 0), 3)
    # Draw text on frame.
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # If we press 'q', end script.
    if key == ord('q'):
        break

camera.release()
cv2.DestroyAllWindows()