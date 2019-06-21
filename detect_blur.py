import argparse
import cv2

def variance_of_laplacian(image):
    '''
    Computes the Laplacian of a single channel image and returns the focus measure (variance of the Laplacian).
    '''
    return cv2.Laplacian(image, cv2.CV_64F).var()

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image.')
ap.add_argument('-t', '--threshold', type=float, default=100.0, help='Focus measures that fall below this value will be labeled as blurry.')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fm = variance_of_laplacian(gray)

text = 'Not blurry.'
if fm < args['threshold']:
    text = 'Blurry.'

# Third to last arg controls size of font and last arg controls thickness of font.
cv2.putText(image, f'{text}: {fm}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.imshow('Image', image)
cv2.waitKey(0)