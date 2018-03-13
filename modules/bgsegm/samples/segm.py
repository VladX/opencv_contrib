import numpy as np
import cv2

cap = cv2.VideoCapture(0)
bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    mask = bgs.apply(frame)
    bg = bgs.getBackgroundImage()
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('bg', bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
