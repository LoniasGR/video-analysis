import cv2
import numpy as np


frames = [[[]]]
# video.mp4: input video
cap = cv2.VideoCapture("Part2/Team30.mp4")

framesCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = np.empty((frameHeight//2, frameWidth//2, framesCount))


for cnt in range(framesCount):
  ret, frame = cap.read()

  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  width = int(grayFrame.shape[1] * 0.5)
  height = int(grayFrame.shape[0] * 0.5)
  

  frames[:,:,cnt] = cv2.resize(grayFrame, (width, height))

cap.release()

print(frames.shape)