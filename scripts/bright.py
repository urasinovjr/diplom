import cv2
import numpy as np

def is_room_bright(image, threshold=100):
    """
    Check if the room is bright based on the image brightness.
    :param image_path: path to the image file
    :param threshold: brightness threshold (default is 100)
    :return: True if the room is bright, False otherwise
    """

    if image is None:
        print("Error: Unable to read the image")
        return False

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness
    brightness = np.mean(gray_image)

    print("Average brightness:", brightness)

    # Check if the average brightness exceeds the threshold
    if brightness > threshold:
        return True
    else:
        return False

