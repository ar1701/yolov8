import ultralytics
import cv2 
import numpy as np

image = cv2.imread('images.jpeg')
keypoints = detect_keypoints(image)

wrist_keypoint = keypoints[0]
elbow_keypoint = keypoints[1]
shoulder_keypoint = keypoints[2]

arm_vector = elbow_keypoint - shoulder_keypoint
forearm_vector = wrist_keypoint - elbow_keypoint

angle = np.arccos(np.dot(arm_vector, forearm_vector) / (np.linalg.norm(arm_vector) * np.linalg.norm(forearm_vector)))
angle_in_degrees = np.degrees(angle)

print("Arm angle:", angle_in_degrees)

elbow_keypoint = keypoints[1]
shoulder_keypoint = keypoints[2]
upper_arm_keypoint = keypoints[3]

upper_arm_vector = shoulder_keypoint - upper_arm_keypoint
forearm_vector = elbow_keypoint - upper_arm_keypoint
biceps_vector = elbow_keypoint - shoulder_keypoint

upper_arm_angle = np.arccos(np.dot(upper_arm_vector, forearm_vector) / (np.linalg.norm(upper_arm_vector) * np.linalg.norm(forearm_vector)))
biceps_angle = np.arccos(np.dot(biceps_vector, forearm_vector) / (np.linalg.norm(biceps_vector) * np.linalg.norm(forearm_vector)))

elbow_angle = upper_arm_angle - biceps_angle
elbow_angle_in_degrees = np.degrees(elbow_angle)

print("Elbow angle:", elbow_angle_in_degrees)
