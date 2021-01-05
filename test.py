
import subprocess
import pickle
import time
import cv2
import os

from faced import FaceDetector
from faced.utils import annotate_image

face_detector = FaceDetector()
#
# img = cv2.imread('foto.jpg')
# rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#
# # Receives RGB numpy image (HxWxC) and
# # returns (x_center, y_center, width, height, prob) tuples.
# bboxes = face_detector.predict(rgb_img, 0.85)
#
# # Use this utils function to annotate the image.
# ann_img = annotate_image(img, bboxes)
#
# # Show the image
# cv2.imshow('image',ann_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('2.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

    # Receives RGB numpy image (HxWxC) and
    # returns (x_center, y_center, width, height, prob) tuples.
    bboxes = face_detector.predict(rgb_img, 0.85)
    print('===', ret, bboxes)

    # Use this utils function to annotate the image.
    ann_img = annotate_image(frame, bboxes)

    # Display the resulting frame
    cv2.imshow('frame', ann_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
