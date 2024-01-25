import cv2
import os
import tensorflow as tf
import keras
import numpy as np
import glob
import re
import csv



optimizers = ["adam", "adamax", "nadam", "rmsprop", "sgd"]
batch_sizes = [8, 16, 32]
epochs = [100, 200]

# model = keras.models.load_model("stupid-model-500-adamax-1.keras")
directory = "Dataset/Train/Grids/*.png"

file_dir = "Dataset/Train/Grid_labels.csv"
images = sorted(glob.glob(directory), key=lambda x:float(re.findall("(\d+)",x)[0]))

model_1 =  keras.models.load_model("jupyter-model.keras")
# model_2 =  keras.models.load_model("stupid-adam-32-500-model.keras")

label_3 = []
label_4 = []

labels = []
with open(file_dir, 'r') as file:
    reader = csv.reader(file)
    for index, lines in enumerate(reader):
        if index == 0 :
            continue
        labels.append(lines[1:10])
class_names = ["Blank", "Cross", "Zeroes"]

grid_mapper = {
    "0": "Cross",
    "1": "Zeroes",
    "2": "Blank"
}

label_mapper = [2, 0, 1]

start_coord = [60, 145]
width = 110
height = 110
random_num = 10000
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
            # predictions.append([class_names[np.argmax(score  )], grid_mapper[labels[index][3*i +j]]])
            # cv2.imshow("image", cropped_img)

            # print(index+1, class_names[np.argmax(score)], grid_mapper[labels[index][3*i + j]])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            prediction.append(label_mapper[np.argmax(score)])
            if class_names[np.argmax(score)] == grid_mapper[labels[index][3*i+j]]:
                correct += 1
            else:
                incorrect += 1
                # cv2.imshow("image", cropped_img)
                # print(index, 3*i+j, class_names[np.argmax(score)], grid_mapper[labels[index][3*i + j]])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
            start_coord[1] += height + 15
            random_num += 1
        
        start_coord[1] = 145
        start_coord[0] += height + 15
    label_3.append(prediction)

print(correct/counter)
exit()
correct = 0
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


            prediction_model = model_2.predict(resized)
            score = tf.nn.softmax(prediction_model[0])
            # predictions.append([class_names[np.argmax(score  )], grid_mapper[labels[index][3*i +j]]])
            # cv2.imshow("image", cropped_img)

            # print(index+1, class_names[np.argmax(score)], grid_mapper[labels[index][3*i + j]])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            prediction.append(label_mapper[np.argmax(score)])
            if class_names[np.argmax(score)] == grid_mapper[labels[index][3*i+j]]:
                correct += 1
            else:
                incorrect += 1
                # cv2.imshow("image", cropped_img)
                # print(index, 3*i+j, class_names[np.argmax(score)], grid_mapper[labels[index][3*i + j]])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
            start_coord[1] += height + 15
            random_num += 1
        
        start_coord[1] = 145
        start_coord[0] += height + 15
    label_4.append(prediction)

print(correct/counter)


with open("labels_1.csv", "w") as file:
    reader = csv.writer(file)
    reader.writerows(label_3)

with open("labels_2.csv", "w") as file:
    reader = csv.writer(file)
    reader.writerows(label_4)


# print(predictions)
# cv2.imshow("cropped", cropped_img)
# cv2.waitKey(0)



# cv2.destroyAllWindows()
# img = cv2.imread("Dataset/Train/Cross/181.png")
# cv2.imshow("image", img)
# cv2.waitKey(0)