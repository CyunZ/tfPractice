import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('MyModel.keras')
# model.summary()

for i in range(5):
    img = cv2.imread('uu'+str(i+1)+'.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(96,96))
    img = img.reshape((-1,96,96,1))
    result = model(img)
    print(result.numpy())
    