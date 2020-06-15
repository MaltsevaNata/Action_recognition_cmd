import cv2
import numpy as np
import os

def get_images(video):
    images = []
    cap = cv2.VideoCapture(video)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = 4
    n = int(total_frames//step)
    for i in range(n):
        # here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
        cap.set(1, i * step)
        success, image = cap.read()
        images.append(image)
    cap.release()
    return images