import csv
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

''' File này dùng để predict trên hết folder test để đưa ra ảnh accuracy
Note: Trước khi chạy file nhớ đổi tên file csv, vì khi chạy nó sẽ ghi đè
'''
### Load Model ###

model_dir = 'model/2022-10-31/rice_model_50.h5' # Đường dẫn model

# Khởi tạo thông số ảnh và model
img_height = 100
img_width = 200
num_classes = 6

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.load_weights(model_dir)
class_names = ['BC-15', 'Huongthom', 'Nep87', 'Q5', 'Thien_uu', 'Xi23']

# Đọc folder ảnh cần predict
test_set_dir = 'data/rice_test'

# Header cho file csv
header = 'path y_true y_predict'
header = header.split()

file = open('test_path_and_y2.csv', 'w') # Đổi tên chỗ này, tên mới sẽ tạo file mới, tên cũ thì ghi đè
with file:
    writer = csv.writer(file)
    writer.writerow(header)
data = []

# Bắt đầu đọc từng folder, rồi từng folder lại đọc tiếp từng ảnh, y_true là tên folder, y_predict là từ model
for rice_folder in os.listdir(test_set_dir):
    rice_folder_dir = os.path.join(test_set_dir, rice_folder) # Đường dẫn folder gạo
    print('rice_folder', rice_folder)
    for rice_img in os.listdir(rice_folder_dir):
        rice_img_dir = os.path.join(rice_folder_dir, rice_img) # Đường dẫn ảnh hạt gạo

        y_true = str(rice_folder) # y_true là tên folder

        img = tf.keras.utils.load_img(
            rice_img_dir, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        y_predict = str(class_names[np.argmax(score)]) # y_predict của model

        # Mở file ra rồi chạy lại
        data = [rice_img_dir, y_true, y_predict]
        file = open('test_path_and_y.csv', 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(data)

