import ultralytics
import cv2
import numpy as np

def calculate_arm_angle(image, keypoints):
    # Extract the keypoints for the wrist, elbow, and shoulder
    wrist_keypoint = keypoints[0]
    elbow_keypoint = keypoints[1]
    shoulder_keypoint = keypoints[2]

    # Calculate the vectors for the arm and forearm
    arm_vector = elbow_keypoint - shoulder_keypoint
    forearm_vector = wrist_keypoint - elbow_keypoint

    # Calculate the angle between the arm and forearm vectors
    angle = np.arccos(np.dot(arm_vector, forearm_vector) / (np.linalg.norm(arm_vector) * np.linalg.norm(forearm_vector)))

    # Convert the angle to degrees
    angle_in_degrees = np.degrees(angle)

    return angle_in_degrees

def calculate_elbow_angle(image, keypoints):
    # Extract the keypoints for the elbow, shoulder, and upper arm
    elbow_keypoint = keypoints[1]
    shoulder_keypoint = keypoints[2]
    upper_arm_keypoint = keypoints[3]

    # Calculate the vectors for the upper arm, forearm, and biceps
    upper_arm_vector = shoulder_keypoint - upper_arm_keypoint
    forearm_vector = elbow_keypoint - upper_arm_keypoint
    biceps_vector = elbow_keypoint - shoulder_keypoint

    # Calculate the angle between the upper arm and forearm vectors
    upper_arm_angle = np.arccos(np.dot(upper_arm_vector, forearm_vector) / (np.linalg.norm(upper_arm_vector) * np.linalg.norm(forearm_vector)))

    # Calculate the angle between the biceps and forearm vectors
    biceps_angle = np.arccos(np.dot(biceps_vector, forearm_vector) / (np.linalg.norm(biceps_vector) * np.linalg.norm(forearm_vector)))

    # Calculate the elbow angle as the difference between the upper arm and biceps angles
    elbow_angle = upper_arm_angle - biceps_angle

    # Convert the angle to degrees
    elbow_angle_in_degrees = np.degrees(elbow_angle)

    return elbow_angle_in_degrees
