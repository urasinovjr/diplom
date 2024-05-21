import cv2
import time
import imutils
import numpy as np
from fer import FER
import csv
import os
import caffe


from imutils.video import FPS
from imutils.video import VideoStream

from yoloutils import *
from posenet import PoseNetDetector 
from sitting_check import is_person_sitting, process_output
from bright import is_room_bright

labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

print('[Status] Loading Model...')
model = caffe.Net("models\yolov3.prototxt", "models\yolov3.caffemodel", caffe.TEST)

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

        inp_dim = 416, 416
        img = img_prepare(frame, inp_dim)

        model.blobs['data'].data[:] = img
        output = model.forward()

        rects = rects_prepare(output)
        mapping = get_classname_mapping(args.classfile)

        scaling_factor = min(1, 416 / frame.shape[1])
        for pt1, pt2, cls, prob in rects:
            pt1[0] -= (416 - scaling_factor*frame.shape[1])/2
            pt2[0] -= (416 - scaling_factor*frame.shape[1])/2
            pt1[1] -= (416 - scaling_factor*frame.shape[0])/2
            pt2[1] -= (416 - scaling_factor*frame.shape[0])/2

            pt1[0] = np.clip(int(pt1[0] / scaling_factor), a_min=0, a_max=frame.shape[1])
            pt2[0] = np.clip(int(pt2[0] / scaling_factor), a_min=0, a_max=frame.shape[1])
            pt1[1] = np.clip(int(pt1[1] / scaling_factor), a_min=0, a_max=frame.shape[1])
            pt2[1] = np.clip(int(pt2[1] / scaling_factor), a_min=0, a_max=frame.shape[1])

            label = "{}:{:.2f}".format(mapping[cls], prob)
            color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

            cv2.rectangle(frame, tuple(pt1), tuple(pt2), color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
            cv2.rectangle(frame, tuple(pt1), tuple(pt2), color, -1)
            cv2.putText(frame, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                        cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)



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
