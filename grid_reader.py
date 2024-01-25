import cv2
import os
import tensorflow as tf
import keras
import numpy as np
import glob
import re
import csv


directory = "Dataset/Test/*.png"

images = sorted(glob.glob(directory), key=lambda x:float(re.findall("(\d+)",x)[0]))

model_1 =  keras.models.load_model("../Jupyter/jupyter-model-4.keras")

label = []

class_names = ["Blank", "Cross", "Zeroes"]

grid_mapper = {
    "0": "Cross",
    "1": "Zeroes",
    "2": "Blank"
}

label_mapper = ["2", "0", "1"]

start_coord = [60, 145]
width = 110
height = 110
predictions = []
correct = 0
incorrect = 0
counter = 1
for index, image in enumerate(images):
    start_coord = [60, 145]
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    prediction = []
    for i in range(3):
        for j in range(3):
            counter += 1
            cropped_img = img[start_coord[0]:start_coord[0]+width, start_coord[1]:start_coord[1]+height]
            resized = cv2.resize(cropped_img, (28, 28))
            
            resized = tf.expand_dims(resized, 0)


            prediction_model = model_1.predict(resized)
            score = tf.nn.softmax(prediction_model[0])

            prediction.append(label_mapper[np.argmax(score)])

            start_coord[1] += height + 15
        
        start_coord[1] = 145
        start_coord[0] += height + 15
    row = [image.split('\\')[-1].split('.')[0]]
    for p in prediction:
        row.append(p)
    print(row)
    label.append(row)

def display(array):
    for e in array:
        print(e)
print("*"*100)
display(label)

with open("labels(2).csv", "w", newline="") as file:
    reader = csv.writer(file)
    reader.writerows(label)
