#!/usr/bin/python3

import mediapipe as mp

mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Setup Pose function for video.
pose_frame = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=2)