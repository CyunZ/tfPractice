import tensorflow as tf
import numpy as np
import cv2 
from PIL import Image


with np.load('GDataSet.npz') as data:
    imgs = data['imgs']
    labels = data['labels']

train_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(96,96,3)) )
model.add( tf.keras.layers.MaxPooling2D((2,2)))
model.add( tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add( tf.keras.layers.MaxPooling2D((2,2)))
model.add( tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add( tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(64,activation='relu') )
model.add( tf.keras.layers.Dense(6) )

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
    )

model.fit(train_dataset, epochs=10)