import cv2
import os
# import tensorflow as tf
# import keras
import numpy as np
import glob
import re
import csv



# model = keras.models.load_model("stupid-model.keras")
directory = "Dataset/Test/*.png"

file_dir = "ICG.csv"
images = sorted(glob.glob(directory), key=lambda x:float(re.findall("(\d+)",x)[0]))
save_dir = "New_Train/"

labels = []
with open(file_dir, 'r') as file:
    reader = csv.reader(file)
    for index, lines in enumerate(reader):
        labels.append(lines)

class_names = ["Cross", "Zero"]
grid_mapper = {
    "0": "Cross",
    "1": "Zeroes",
    "2": "Blank"
}

start_coord = [60, 145]
width = 110
height = 110
random_num = 100000
predictions = []
for index, image in enumerate(images):
    start_coord = [60, 145]
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if(len(labels[index]) == 0):
        continue
    for i in range(3):
        for j in range(3):
                
            cropped_img = img[start_coord[0]:start_coord[0]+width, start_coord[1]:start_coord[1]+height]
            resized = cv2.resize(cropped_img, (28, 28))
            cv2.imwrite(save_dir+grid_mapper[labels[index][3*i + j]]+"/"+str(random_num)+".png", resized)
            
            # resized = tf.expand_dims(resized, 0)


            # prediction = model.predict(resized)
            # score = tf.nn.softmax(prediction[0])
            # predictions.append(class_names[np.argmax(score  )])

            # print(
            #     "This image most likely belongs to {} with a {:.2f} percent confidence."
            #     .format(class_names[np.argmax(score)], 100 * np.max(score))
            # )
            start_coord[1] += height + 15
            random_num += 1
        start_coord[1] = 145
        start_coord[0] += height + 15
    if index == 49:
        break

# print(predictions)
# cv2.imshow("cropped", cropped_img)
# cv2.waitKey(0)



# cv2.destroyAllWindows()
# img = cv2.imread("Dataset/Train/Cross/181.png")
# cv2.imshow("image", img)
# cv2.waitKey(0)