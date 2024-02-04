import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.save('handwritten.model')

# model = tf.keras.models.load_model('handwritten.model')

for image_number in range(0, 10):
    filename = f"digits/digit{image_number}.png"
    if os.path.isfile(filename):
        try:
            img = cv2.imread(filename)[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a: {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error processing image {image_number}: {e}")
    else:
        print(f"File {filename} not found.")
