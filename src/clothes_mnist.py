# -*- coding: utf-8 -*-
"""
Created on Mon May 4 2020

@author: Fernando Roldán Zafra & Lidia Sanchez Mérida
"""

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score

import time
import matplotlib.pyplot as plt

batch_size = 256
num_classes = 10
epochs = 30


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
                                  
test_images = test_images.reshape((10000, 28*28))                                  
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = Sequential()
network.add(Dense(256, activation='relu', input_shape=(28*28,)))
network.add(Dropout(rate=0.5))
network.add(Dense(num_classes, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.summary()
start_time = time.time()
history = network.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(test_images, test_labels))
elapsed_time = time.time() - start_time

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show() 

score = network.evaluate(test_images, test_labels, verbose=0)
test_predicted = network.predict(test_images)
test_labels = test_labels.argmax(axis=-1)
test_predicted = test_predicted.argmax(axis=-1)
kappa = cohen_kappa_score(test_labels, test_predicted)

print('-----------------Test-----------------')
print('test_acc:', score[1])
print('Test fails', "%0.2f" %(100 - score[1]*100))
print('Time: ', "%0.2f" %elapsed_time)
print('Cohens kappa: ', "%0.2f" %kappa)