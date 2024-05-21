import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class PoseNetDetector:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def process_image(self, image):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data = np.expand_dims(cv2.resize(image, (input_shape[1], input_shape[2])), axis=0)

        input_data = np.uint8(input_data)

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        return output_data
    
def process_output(output_data):
    keypoints = output_data[0, 0, :, :]
    processed_keypoints = []
    for keypoint in keypoints:
        y, x, confidence = keypoint
        processed_keypoints.append({'y': y, 'x': x, 'confidence': confidence})
    return processed_keypoints

def draw_skeleton(image, keypoints):
    connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for a, b in connections:
        y1, x1 = int(keypoints[a][1] * image.shape[0]), int(keypoints[a][0] * image.shape[1])
        y2, x2 = int(keypoints[b][1] * image.shape[0]), int(keypoints[b][0] * image.shape[1])
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
def draw_poses(image, output_data):
    for pose_data in output_data:
        keypoints = process_output(pose_data)
        print(keypoints)
        if keypoints is not None:
            for keypoint in keypoints:
                if keypoint['confidence'] > 0.2: 
                    cv2.circle(image, (int(keypoint['x'] * image.shape[1]), int(keypoint['y'] * image.shape[0])), 5, (0, 255, 0), -1)

def detect_poses(input_folder, output_folder, model_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    posenet_detector = PoseNetDetector(model_path)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            poses = posenet_detector.process_image(image)

            # Рисуем обнаруженные позы на изображении
            draw_poses(image, poses)

            # Сохраняем изображение с выделенными позами
            cv2.imwrite(os.path.join(output_folder, filename), image)

            # Выводим информацию о прогрессе
            print(f"Processed image: {filename}")

if __name__ == "__main__":
    input_folder = 'data/processed'
    output_folder = 'data/annotations/poses'
    model_path = 'models/4.tflite'

    detect_poses(input_folder, output_folder, model_path)
