import cv2
import mediapipe as mp
import csv
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

output_file = "pose_data.csv"

if not os.path.exists(output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Label"]
        for landmark in mp_pose.PoseLandmark:
            header += [f"{landmark.name}_x", f"{landmark.name}_y", f"{landmark.name}_z", f"{landmark.name}_visibility"]
        writer.writerow(header)

def save_pose_data(landmarks, label):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [label]
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        writer.writerow(row)
    print(f"Pose data saved for label: {label}")

cap = cv2.VideoCapture(0)  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        instructions = "Press 's' to save data, 'q' to quit."
        cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Pose Data Capture', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and results.pose_landmarks:
            label = input("Enter label for the current pose (e.g., walking, running, sitting): ").strip()
            if label:
                save_pose_data(results.pose_landmarks.landmark, label)

cap.release()
cv2.destroyAllWindows()