import cv2
import time
import imutils
import argparse
import numpy as np
from fer import FER
import csv
import os

from imutils.video import FPS
from imutils.video import VideoStream

from posenet import PoseNetDetector 
from sitting_check import is_person_sitting, process_output
from bright import is_room_bright

labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe("models\SSD_MobileNet_prototxt.txt", "models\SSD_MobileNet.caffemodel")

posenet_detector = PoseNetDetector("models/4.tflite")
detector = FER(mtcnn=True)

#Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Открытие файла CSV для записи
csv_filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
with open(csv_filename, mode='w', newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Datetime', 'Room Brightness', 'Person Status', 'Dominant Emotion', 'Person Detected'])

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        nn.setInput(blob)
        detections = nn.forward()

        poses = posenet_detector.process_image(frame)
        for pose_data in poses:
            keypoints = process_output(pose_data)
            if keypoints is not None:
                if is_person_sitting(keypoints):
                    person_status = "Сидит"
                else:
                    person_status = "Стоит"

        room_brightness = "Ярко освещена" if is_room_bright(frame) else "Темновата"
        emotions = detector.detect_emotions(frame)
        try:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        except:
            dominant_emotion = "Нет лица"
        
        person_detected = "Да" if "person" in labels else "Нет"

        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), room_brightness, person_status, dominant_emotion, person_detected])

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >  0.7:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
