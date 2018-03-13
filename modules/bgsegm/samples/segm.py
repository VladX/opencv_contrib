import sys
sys.path.insert(0, '../../../../build/lib/python3/')
import numpy as np
import cv2
import torch
from torch.utils.serialization import load_lua
from skimage import io, transform

print('OpenCV Version:', cv2.__version__)

model = load_lua(sys.argv[1])

def forward(img):
    H, W = img.shape[0:2]
    img = np.float32(img) / 255.0
    img = cv2.resize(img, (256, 256))
    img = np.transpose(img, [2, 0, 1])
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    result = model.forward(img)
    mask = np.transpose(result[0].numpy(), [1, 2, 0])
    mask = mask[:,:,1]
    return cv2.resize(mask, (W, H))

cap = cv2.VideoCapture(0)
bgs = cv2.bgsegm.createBackgroundSubtractorGSOC(propagationRate=0.1)

while True:
    ret, frame = cap.read()
    emask = forward(frame)
    emask = np.uint8((emask > 0.5) * 255)
    mask = bgs.apply_with_mask(frame, emask)
    bg = bgs.getBackgroundImage()
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('exponet', emask)
    cv2.imshow('bg', bg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
