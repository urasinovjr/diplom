import os
import imgaug.augmenters as iaa
import cv2
import random

def augment_images(input_folder, output_folder, augment_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Случайное отражение по горизонтали
        iaa.Affine(rotate=(-10, 10)),  # Случайный поворот на угол от -10 до 10 градусов
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Случайное размытие
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Добавление случайного гауссовского шума
    ])

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            for i in range(augment_count):
                augmented_image = seq(image=image)
                output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_aug_{i+1}.jpg")
                cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    input_folder = "data/raw"
    output_folder = 'data/processed' 
    augment_count = 2

    augment_images(input_folder, output_folder, augment_count)
