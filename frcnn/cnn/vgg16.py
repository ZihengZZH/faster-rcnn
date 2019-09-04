"""
VGG16 model for Keras.
---
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Convolution2D, MaxPooling2D
from keras import backend as K


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length/16
    return get_output_length(width), get_output_length(height)    


def nn_base(input_tensor=None, trainable=False):
    """
    Neural Net Base of VGG16 (key to use)
    ---
    depth64     (3x3 conv): conv1_1 / conv1_2
    depth128    (3x3 conv): conv2_1 / conv2_2
    depth256    (3x3 conv): conv3_1 / conv3_2 / conv3_3
    depth512    (3x3 conv): conv4_1 / conv4_2 / conv4_3
    depth512    (3x3 conv): conv5_1 / conv5_2 / conv5_3
    size4096     fully:     fc1 / fc2
    size1000     softmax
    """
    # Determine proper input shape
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

