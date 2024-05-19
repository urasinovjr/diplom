import cv2
import os

class VideoToFrames:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)

    def extract_frames(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % 60 == 0: 
                cv2.imwrite(os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg"), frame)
            frame_count += 1

        self.cap.release()

# Usage
video_path = "data/test.mp4"
output_dir = "data/raw"
extractor = VideoToFrames(video_path, output_dir)
extractor.extract_frames()