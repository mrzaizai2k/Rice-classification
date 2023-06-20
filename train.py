import datetime
import pathlib

import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''File này dùng để train model'''

### Đọc ảnh và đường dẫn ###

datetime_object = datetime.date.today() # Ngày giờ tiện cho việc lưu tên model và ảnh
data_dir = pathlib.Path('data/rice_train') #Đường dẫn đến train dataset
# Tỉ lệ phân chia train/valid/test được viết trong file 'data/rice.txt

image_count = len(list(data_dir.glob('*/*.png')))  # Đếm tổng số ảnh có trong rice dataset
print(image_count)

# Coi thử 1 ảnh trong folder Huongthom
Huongthom = list(data_dir.glob('Huongthom/*'))
img = PIL.Image.open(str(Huongthom[0]))
# img.show()
# Muốn coi ảnh thì tắt # im.show

### Tạo Dataset ###

batch_size = 32
img_height = 100
img_width = 200
# Tỉ lệ ảnh hạt gạo thường là 2:1 nên chọn ảnh đầu vào là 200x100 pixels

# Chia tập train và valid
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Này như data loader để chuẩn bị cho việc nạp ảnh vào model
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

### Chuẩn hóa ảnh

# Chuyển giá trị ảnh về 0-1
normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

# Data Augmentation
# Tăng số lượng data để tăng accuracy, Chỉ dùng khi train chứ không phải test,
# lưu ý 2 cấu trúc model lúc train và test khác nhau chỗ này
# Các phương pháp augmentation gồm flip, zoom và rotate ảnh
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
# plt.show()

# Khởi tạo model
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])


model.summary()


early_stopping_patience = 7 # Cho việc dừng sớm nếu train quá 7 epoch mà ko đạt KQ tốt hơn

# Cho learning rate giảm dẫn theo epoch, để tối ưu việc đạt cực trị
def scheduler(epoch):
    if epoch <= 15:
        return 1e-3
    elif 15 < epoch <= 20:
        return 1e-4
    else:
        return 1e-5


my_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler), # Thay đổi learning rate
    tf.keras.callbacks.ModelCheckpoint(filepath='model/' + str(datetime_object) + '/rice_model_{epoch:02d}.h5',
                                       save_freq='epoch',
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True,
                                       period=5), # Lưu mỗi 5 epoch nên KQ tốt hơn, sợ cúp điện thì nó ko lưu
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        # Dừng train sớm nếu KQ mãi ko tiến triển
    )
]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Train model
epochs = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=my_callbacks,
)

# Vẽ accuracy và loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot( acc, label='Training Accuracy')
plt.plot( val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot( loss, label='Training Loss')
plt.plot( val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('model\{}\Training and Validation Loss.png'.format(datetime_object))
plt.show()
