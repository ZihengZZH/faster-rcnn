"""
ResNet50 model in Keras.
---
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Add, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D
from keras.layers import TimeDistributed

from keras import backend as K

from frcnn.fixed_batchnorm import FixedBatchNormalization

batchnorm_axis = 3


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero padding
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length
    return get_output_length(width), get_output_length(height) 


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """
    Identity block within ResNet
    """
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    batchnorm_name_base = 'batchnorm' + str(stage) + block + '_branch'

    x = Convolution2D(num_filter1, (1, 1), 
                        name=conv_name_base+'2a', 
                        trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(num_filter2, (kernel_size, kernel_size), 
                        padding='same', 
                        name=conv_name_base+'2b', 
                        trainable=trainable)(x)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(num_filter3, (1, 1), 
                        name=conv_name_base+'2c', 
                        trainable=trainable)(x)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_timedist(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """
    Indentity block (time distributed) within ResNet
    """
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    batchnorm_name_base = 'batchnorm' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(num_filter1, (1, 1), 
                                    trainable=trainable, 
                                    kernel_initializer='normal'), 
                        name=conv_name_base+'2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(num_filter2, (kernel_size, kernel_size), 
                                    trainable=trainable, 
                                    kernel_initializer='normal',
                                    padding='same'), 
                        name=conv_name_base+'2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(num_filter3, (1, 1), 
                                    trainable=trainable, 
                                    kernel_initializer='normal'), 
                        name=conv_name_base+'2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """
    Conv block within ResNet
    """
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    batchnorm_name_base = 'batchnorm' + str(stage) + block + '_branch'

    x = Convolution2D(num_filter1, (1, 1), 
                        strides=strides, 
                        name=conv_name_base+'2a', 
                        trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(num_filter2, (kernel_size, kernel_size), 
                        padding='same', 
                        name=conv_name_base+'2b', 
                        trainable=trainable)(x)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(num_filter3, (1, 1), 
                        name=conv_name_base+'2c', 
                        trainable=trainable)(x)
    x = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base+'2c')(x)

    shortcut = Convolution2D(num_filter3, (1, 1), strides=strides, 
                    name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=batchnorm_axis, name=batchnorm_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_timedist(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    """
    Conv block (time distributed) within ResNet
    """
    num_filter1, num_filter2, num_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    batchnorm_name_base = 'batchnorm' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(num_filter1, (1, 1), 
                                    strides=strides, 
                                    trainable=trainable, 
                                    kernel_initializer='normal'), 
                        input_shape=input_shape, 
                        name=conv_name_base+'2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(num_filter2, (kernel_size, kernel_size), 
                                    padding='same', 
                                    trainable=trainable, 
                                    kernel_initializer='normal'), 
                        name=conv_name_base+'2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(num_filter3, (1, 1), 
                                    kernel_initializer='normal'), 
                        name=conv_name_base+'2c', 
                        trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'2c')(x)

    shortcut = TimeDistributed(Convolution2D(num_filter3, (1, 1), 
                                            strides=strides, 
                                            trainable=trainable, 
                                            kernel_initializer='normal'), 
                                name=conv_name_base+'1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=batchnorm_axis), name=batchnorm_name_base+'1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def nn_base(input_tensor=None, trainable=False):
    """
    Neural Net Base of ResNet (key to use)
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

    batchnorm_axis = 3

    # Stage 1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=batchnorm_axis, name='batchnorm_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

    return x


def classifier_layers(x, input_shape, trainable=False):
    """
    Additional Classifier Layer of ResNet
    """
    # Stage 5
    x = conv_block_timedist(x, 3, [512, 512, 2048], stage=5, block='a', 
                            input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_timedist(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_timedist(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x

