from fer import FER
import cv2

detector = FER(mtcnn=True)

image_path = 'path_to_your_image.jpg'
image = cv2.imread(r"data\test.jpg")

emotions = detector.detect_emotions(image)

print(emotions)
dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
dominant_emotion_confidence = emotions[0]['emotions'][dominant_emotion]


x, y, w, h = emotions[0]['box']
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.putText(image, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
