from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

vgg_name = 'vgg19'

batch_size = 128
num_class = 10
epochs = 20

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("train samples", X_train.shape)
print("test samples", X_train.shape)

y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

if vgg_name == 'vgg16':
    from frcnn.cnn import vgg16 as vgg
else:
    from frcnn.cnn import vgg19 as vgg

input_tensor = Input(shape=input_shape)
x = vgg.nn_base(input_tensor=input_tensor, trainable=True) 
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x) 
x = Dense(4096, activation='relu', name='fc2')(x) 
x = Dense(num_class, activation='softmax', name='class')(x)         
model = Model(input_tensor, x)

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])
plot_model(model, to_file='./images/mnist_%s.png' % vgg_name)

model.fit(X_train, y_train, batch_size=batch_size,
            epochs=epochs, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print("test loss\t", score[0])
print("test accuracy\t", score[1])


