import datetime
import pathlib

import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet import ResNet50

'''File này để test trên 1 tấm ảnh bất kì trong folder data '''

### Đọc ảnh và đường dẫn ###

image_dir = 'data/rice_test/Thien_uu/DSC6777_idx2.png' # Đường dẫn ảnh cần test
model_dir = 'model/2022-11-04/rice_model_resnet_12.h5' # Đường dẫn model đã train, file h5 tensorflow

# Cài đặt thông số ảnh khi đưa vào model
img_height = 100
img_width = 200
num_classes = 6 # Số loại gạo

# Biến đổi ảnh thành ma trận để phù hợp với đầu vào của model
img = tf.keras.utils.load_img(
    image_dir, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Khởi tạo lại cấu trúc model
resnet = ResNet50(weights= 'imagenet', include_top=False, input_shape= (img_height,img_width,3))

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = "relu")(x)
x = Dropout(0.3)(x)#
x = Dense(256, activation = "relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = resnet.input, outputs = predictions)

# Nạp trọng số của model
model.load_weights(model_dir)
class_names = ['BC-15', 'Huongthom', 'Nep87', 'Q5', 'Thien_uu', 'Xi23']
predictions = model.predict(img_array) # In ra xác suất của loại gạo đó
score = tf.nn.softmax(predictions[0]) # Chuyển xác suất về 0-1

print(
    "Day la gao {} voi {:.2f} % confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
