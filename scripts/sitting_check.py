import cv2
import numpy as np
import os
from posenet import PoseNetDetector 

def is_person_sitting(keypoints):
    """
    Check if a person is sitting based on the detected keypoints.
    :param keypoints: list of detected keypoints
    :return: True if the person is sitting, False otherwise
    """
    left_hip = keypoints[11]  # Левый бедренный сустав
    right_hip = keypoints[12]  # Правый бедренный сустав
    left_knee = keypoints[13]  # Левое колено
    right_knee = keypoints[14]  # Правое колено

    # Для определения сидящего положения проверяем, находятся ли бедра ниже коленей
    if left_hip['y'] > left_knee['y'] and right_hip['y'] > right_knee['y']:
        return True
    else:
        return False
    
def process_output(output_data):
    keypoints = output_data[0, 0, :, :]
    processed_keypoints = []
    for keypoint in keypoints:
        y, x, confidence = keypoint
        processed_keypoints.append({'y': y, 'x': x, 'confidence': confidence})
    return processed_keypoints


def main():
    input_folder = 'data/processed'
    output_folder = 'data/annotations/poses'
    model_path = 'models/4.tflite'

    posenet_detector = PoseNetDetector(model_path)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            poses = posenet_detector.process_image(image)

            for pose_data in poses:
                keypoints = process_output(pose_data)
                if keypoints is not None:
                    # Оцениваем, сидит ли человек на основе обнаруженных ключевых точек
                    if is_person_sitting(keypoints):
                        print("Человек сидит")
                    else:
                        print("Человек стоит")            # Выводим информацию о прогрессе
                    print(f"Processed image: {filename}")

if __name__ == "__main__":
    main()
