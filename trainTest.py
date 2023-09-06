import tensorflow as tf
import numpy as np



with np.load('TrainDataSet.npz') as data:
    imgs = data['imgs']
    labels = data['labels']
train_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

with np.load('TestDataSet.npz') as data:
    imgs = data['imgs']
    labels = data['labels'] 
test_dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

BATCH_SIZE = 20
SHUFFLE_BUFFER_SIZE = 30

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE) .batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Conv2D(32,(3,3),(2,2),activation='relu',input_shape=(96,96,3)) )
model.add( tf.keras.layers.AveragePooling2D((2,2)))
model.add( tf.keras.layers.Conv2D(64,(3,3),(2,2),activation='relu'))
model.add( tf.keras.layers.AveragePooling2D((2,2)))
model.add( tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add( tf.keras.layers.AveragePooling2D((2,2)))
model.add( tf.keras.layers.Flatten())
model.add( tf.keras.layers.Dense(6) )


model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
    )

model.fit(train_dataset, epochs=10)
print('测试')
model.evaluate(test_dataset)



