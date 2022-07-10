import cv2
import os

vidcap = cv2.VideoCapture('VID.mp4')
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite(f"new_imgs/frame{count}.jpg", image)     # save frame as JPEG file      
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

print(count)