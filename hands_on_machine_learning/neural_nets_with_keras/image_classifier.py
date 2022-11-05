import matplotlib
import tensorflow as tf
from tensorflow import keras
#
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
#
X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
y_valid, y_train = y_train_full[:5000] / 255, y_train_full[5000:] / 255
# print(y_train)
# array([4, 0, 7, ..., 3, 0, 5], dtype=uint8)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# class_names[y_train[0]]
# 'Coat'
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
"""
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
"""

# model.summary()

# keras.utils.plot_model(model, "./image/fashion_mnist_model.png")

hidden1 = model.layers[1]
# print(hidden1.name)
# 'dense'
# print(model.get_layer(hidden1.name) is hidden1)
# True (is 比较两个对象是否相等而非对象的值)
weights, biases = hidden1.get_weights()
# print(weights)
# print(biases,biases.shape)
# 略
#
model.compile(loss="sparse_catogorical_crossentropy",
              optimizer='sgd', metrics=['accuracy'])
#
history =model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))
# print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
#
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

#
model.evaluate(X_test, y_test)
#
X_new = X_test[:3]
y_proba = model.predict[X_new]
# print(y_proba.round(2))
#
# 略

#
import numpy as np
y_pred = np.argmax(model.predict(X_new), axis=1)
# print(y_pred)
# array([9, 2, 1])
# print(np.array(class_names)[y_pred])
# array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
y_new = y_test[:3]
# print(y_new)
# array([9, 2, 1], dtype=uint8)