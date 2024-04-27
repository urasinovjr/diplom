import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from posenet import PoseNetDetector, process_output

def detect_and_save_poses(input_folder_0, output_folder, model_path, csv_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_keypoints_data = []
    all_labels = []

    posenet_detector = PoseNetDetector(model_path)

    for filename in tqdm(os.listdir(input_folder_0)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder_0, filename)
            image = cv2.imread(image_path)

            poses = posenet_detector.process_image(image)

            for pose_data in poses:
                keypoints = process_output(pose_data)
                all_keypoints_data.append(keypoints)
                all_labels.append(1)

            cv2.imwrite(os.path.join(output_folder, filename), image)

    df = pd.DataFrame(all_keypoints_data, columns=[
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'])

    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    input_folder = 'data/processed'
    output_folder = 'data/annotations/poses'
    model_path = 'models/4.tflite'
    csv_path = "log_pose.csv"

    detect_and_save_poses(input_folder, output_folder, model_path, csv_path)
