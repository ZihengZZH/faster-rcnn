from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from frcnn.cnn import resnet50

batch_size = 256
num_class = 10
epochs = 50

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.stack((X_train, X_train, X_train), axis=3)
X_test = np.stack((X_test, X_test, X_test), axis=3)
input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("train samples", X_train.shape)
print("test samples", X_train.shape)

y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

input_tensor = Input(shape=input_shape)
x = resnet50.nn_base(input_tensor=input_tensor, trainable=True)       
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(num_class, activation='softmax', name='fc')(x)         
model = Model(input_tensor, x)

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='../images/mnist_resnet50.png')

model.fit(X_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_data=(X_test, y_test))

model.save_weights('../weights/mnist_resnet50_b%d_e%d.hdf5' % (batch_size, epochs))

score = model.evaluate(X_test, y_test, verbose=0)
print("test loss\t", score[0])
print("test accuracy\t", score[1])


