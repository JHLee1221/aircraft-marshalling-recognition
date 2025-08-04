#!usr/bin/python3

import cv2
import matplotlib.pyplot as plt
from pose.mpose import *

def detectPose(image, pose, display=True):
    
    # Create copy image
    replica = image.copy()

    # Convert BGR image to RGB image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(RGB_img)

    height, width, _ = image.shape

    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        mp_drawing.draw_landmarks(image=replica, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:

            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

    # Check if the original input image and the resultant image are specified to be displayed.
    if display:

        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(replica[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    else:

        return replica, landmarks
