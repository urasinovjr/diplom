import cv2
import time
import imutils
import numpy as np
from fer import FER
import csv
import os

from imutils.video import FPS
from imutils.video import VideoStream

from posenet import PoseNetDetector, draw_poses
from sitting_check import is_person_sitting, process_output
from bright import is_room_bright

labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe("models\SSD_MobileNet_prototxt.txt", "models\SSD_MobileNet.caffemodel")

posenet_detector = PoseNetDetector("models/4.tflite")
detector = FER(mtcnn=True)

# Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def draw_skeleton(image, keypoints):
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for a, b in connections:
        if a < len(keypoints) and b < len(keypoints):  # Ensure both keypoints exist
            keypoint_a = keypoints[a]
            keypoint_b = keypoints[b]
            confidence_a = keypoint_a.get('confidence', 0)
            confidence_b = keypoint_b.get('confidence', 0)
            # Draw line only if both keypoints have confidence greater than 0.2
            if confidence_a > 0.2 and confidence_b > 0.2:
                x1, y1 = int(keypoint_a['x'] * image.shape[1]), int(keypoint_a['y'] * image.shape[0])
                x2, y2 = int(keypoint_b['x'] * image.shape[1]), int(keypoint_b['y'] * image.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Open CSV file for writing
csv_filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
with open(csv_filename, mode='w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Datetime', 'Room Brightness', 'Person Status', 'Dominant Emotion', 'Person Detected', 'Other Objects'])

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        nn.setInput(blob)
        detections = nn.forward()

        poses = posenet_detector.process_image(frame)
        draw_poses(frame, poses)  # Draw pose keypoints
        for pose_data in poses:
            keypoints = process_output(pose_data)
            draw_skeleton(frame, keypoints)
            
            if keypoints is not None:
                
                if is_person_sitting(keypoints):
                    person_status = "Сидит"
                else:
                    person_status = "Стоит"
                person_detected = "Да"
            else:
                person_status = "Нет человека"


        room_brightness = "Ярко освещена" if is_room_bright(frame) else "Темновата"
        emotions = detector.detect_emotions(frame)
        try:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        except:
            dominant_emotion = "Нет лица"
        
        person_detected = "Нет"
        other_objects = []

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                idx = int(detections[0, 0, i, 1])
                if labels[idx] == "person":
                    person_detected = "Да"
                else:
                    other_objects.append(labels[idx])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), room_brightness, person_status, dominant_emotion, person_detected, ", ".join(other_objects)])

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
