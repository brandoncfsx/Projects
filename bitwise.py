import numpy as np
import cv2

rec_canvas = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(rec_canvas, (25, 25), (275, 275), 255, -1)
cv2.imshow('Rectangle', rec_canvas)
cv2.waitKey(0)

cir_canvas = np.zeros((300, 300), dtype='uint8')
radius = 150
cv2.circle(cir_canvas, (cir_canvas.shape[1] // 2, cir_canvas.shape[0] // 2), radius, 255, -1)
cv2.imshow('Circle', cir_canvas)
cv2.waitKey(0)

bitwiseAnd = cv2.bitwise_and(rec_canvas, cir_canvas)
cv2.imshow('AND', bitwiseAnd)
cv2.waitKey(0)

bitwiseOr = cv2.bitwise_or(rec_canvas, cir_canvas)
cv2.imshow('OR', bitwiseOr)
cv2.waitKey(0)

bitwiseXor = cv2.bitwise_xor(rec_canvas, cir_canvas)
cv2.imshow('XOR', bitwiseXor)
cv2.waitKey(0)

bitwiseNot = cv2.bitwise_not(cir_canvas)
cv2.imshow('NOT', bitwiseNot)
cv2.waitKey(0)