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

'''File này để test trên 1 tấm ảnh bất kì trong folder data'''

### Đọc ảnh và đường dẫn ###

datetime_object = datetime.date.today() # Lấy ngày giờ tiện cho việc lưu tên
image_dir = 'data/rice_test/Thien_uu/DSC6777_idx2.png' # Đường dẫn ảnh cần test
model_dir = 'model/2022-11-05/rice_model_Lenet_27.h5' # Đường dẫn model đã train, file h5 tensorflow

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
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(6, 5, padding='valid', activation='relu'),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2),

    layers.Conv2D(16, 5, padding='valid', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=[2, 2], strides=2),

    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])
# Nạp trọng số của model
model.load_weights(model_dir)
class_names = ['BC-15', 'Huongthom', 'Nep87', 'Q5', 'Thien_uu', 'Xi23']
predictions = model.predict(img_array) # In ra xác suất của loại gạo đó
score = tf.nn.softmax(predictions[0]) # Chuyển xác suất về 0-1

print(
    "Day la gao {} voi {:.2f} % confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
