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

from yoloutils import infer_image, draw_labels_and_boxes

from common.helper import OUTCLS

# Load labels and colors for YOLO
labels = open("models/coco-labels.txt").read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

print('[Status] Loading Model...')
net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights")

# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

posenet_detector = PoseNetDetector("models/4.tflite")
detector = FER(mtcnn=True)

print('[Status] Starting Video Stream...')
count = 0

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

window_name = "Frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

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
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6
        
        dominant_emotion = "Нет лица"

        person_status = "Нет человека"
        person_detected = "Нет"

        other_objects = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                if labels[classids[i]] in OUTCLS:
                    if labels[classids[i]] == "person":
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        cropped_image = frame[y:y+h, x:x+w]

                        poses = posenet_detector.process_image(cropped_image)
                        draw_poses(cropped_image, poses)
                        person_status = "Нет человека"
                        person_detected = "Нет"
                        if poses:
                            for pose_data in poses:
                                keypoints = process_output(pose_data)
                                draw_skeleton(cropped_image, keypoints)
                                
                                if keypoints is not None:
                                    if is_person_sitting(keypoints):
                                        person_status = "Сидит"
                                    else:
                                        person_status = "Стоит"
                                    person_detected = "Да"
                        
                        emotions = detector.detect_emotions(frame)
                        try:
                            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                        except:
                            dominant_emotion = "Нет лица"
                            
                        frame[y:y+h, x:x+w] = cropped_image
                        color = [int(c) for c in colors[classids[i]]]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        text = "{}: {:4f} | {} | {}".format(labels[classids[i]], confidences[i], dominant_emotion, person_status)
                        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        other_objects.append(labels[classids[i]])
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        color = [int(c) for c in colors[classids[i]]]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
                        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        room_brightness = "Ярко освещена" if is_room_bright(frame) else "Слабо освещена"

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
