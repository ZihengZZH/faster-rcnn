"""
Xception V1 model in Keras.
---
# Reference:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import TimeDistributed, Concatenate, Lambda

from keras import backend as K



def get_img_output_length(width, height):
    def get_output_length(input_length):
        # filter_sizes = [3, 3, 3, 3, 3, 3, 3]
        # strides = [2, 1, 2, 1, 2, 2, 2]
        filter_sizes = [3, 3, 3, 3, 3, 3]
        strides = [2, 1, 2, 1, 2, 2]

        assert len(filter_sizes) == len(strides)

        for i in range(len(filter_sizes)):
            input_length = (input_length - filter_sizes[i]) // strides[i] + 1
        return input_length

    return get_output_length(width), get_output_length(height)


def conv2d_batchnorm(x, filters,
                    kernel_size,
                    strides=1,
                    padding='same',
                    activation='relu',
                    use_bias=False,
                    name=None):
    """Utility function to apply conv + batchnorm
    --
    Args:
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters, kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                scale=False,
                                name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def conv2d_batchnorm_timedist(x, filters,
                            kernel_size,
                            strides=1,
                            padding='same',
                            activation='relu',
                            use_bias=False,
                            name=None):
    """Utility function to apply conv + batchnorm + timedist
    --
    Args:
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = TimeDistributed(Conv2D(filters, kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias),
                        name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False),
                            name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block
    --
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    Args:
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    Returns:
        Output tensor for the block.
    Raises:
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    block_name = block_type + '_' + str(block_idx)

    if block_type == 'block35':
        branch_0 = conv2d_batchnorm(x, 32, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm(x, 32, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm(branch_1, 32, 3, name=block_name + '_conv3')
        branch_2 = conv2d_batchnorm(x, 32, 1, name=block_name + '_conv4')
        branch_2 = conv2d_batchnorm(branch_2, 48, 3, name=block_name + '_conv5')
        branch_2 = conv2d_batchnorm(branch_2, 64, 3, name=block_name + '_conv6')
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_batchnorm(x, 192, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm(x, 128, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm(branch_1, 160, [1, 7], name=block_name + '_conv3')
        branch_1 = conv2d_batchnorm(branch_1, 192, [7, 1], name=block_name + '_conv4')
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_batchnorm(x, 192, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm(x, 192, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm(branch_1, 224, [1, 3], name=block_name + '_conv3')
        branch_1 = conv2d_batchnorm(branch_1, 256, [3, 1], name=block_name + '_conv4')
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                        'Expects "block35", "block17" or "block8", '
                        'but got: ' + str(block_type))

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_batchnorm(mixed, K.int_shape(x)[channel_axis], 1,
                        activation=None,
                        use_bias=True,
                        name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                output_shape=K.int_shape(x)[1:],
                arguments={'scale': scale},
                name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def inception_resnet_block_td(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block
    --
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    Args:
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    Returns:
        Output tensor for the block.
    Raises:
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    block_name = block_type + '_' + str(block_idx)

    if block_type == 'block35':
        branch_0 = conv2d_batchnorm_timedist(x, 32, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm_timedist(x, 32, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm_timedist(branch_1, 32, 3, name=block_name + '_conv3')
        branch_2 = conv2d_batchnorm_timedist(x, 32, 1, name=block_name + '_conv4')
        branch_2 = conv2d_batchnorm_timedist(branch_2, 48, 3, name=block_name + '_conv5')
        branch_2 = conv2d_batchnorm_timedist(branch_2, 64, 3, name=block_name + '_conv6')
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_batchnorm_timedist(x, 192, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm_timedist(x, 128, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm_timedist(branch_1, 160, [1, 7], name=block_name + '_conv3')
        branch_1 = conv2d_batchnorm_timedist(branch_1, 192, [7, 1], name=block_name + '_conv4')
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_batchnorm_timedist(x, 192, 1, name=block_name + '_conv1')
        branch_1 = conv2d_batchnorm_timedist(x, 192, 1, name=block_name + '_conv2')
        branch_1 = conv2d_batchnorm_timedist(branch_1, 224, [1, 3], name=block_name + '_conv3')
        branch_1 = conv2d_batchnorm_timedist(branch_1, 256, [3, 1], name=block_name + '_conv4')
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                        'Expects "block35", "block17" or "block8", '
                        'but got: ' + str(block_type))

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 4
    mixed = Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_batchnorm_timedist(mixed, K.int_shape(x)[channel_axis], 1,
                                activation=None,
                                use_bias=True,
                                name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                output_shape=K.int_shape(x)[1:],
                arguments={'scale': scale},
                name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def nn_base(input_tensor=None, trainable=False):

    # Determine proper input shape
    if K.image_data_format() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # Stem block: 35 x 35 x 192
    x = conv2d_batchnorm(img_input, 32, 3, strides=2, padding='valid', name='Stem_block' + '_conv1')
    x = conv2d_batchnorm(x, 32, 3, padding='valid', name='Stem_block' + '_conv2')
    x = conv2d_batchnorm(x, 64, 3, name='Stem_block' + '_conv3')
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_batchnorm(x, 80, 1, padding='valid', name='Stem_block' + '_conv4')
    x = conv2d_batchnorm(x, 192, 3, padding='valid', name='Stem_block' + '_conv5')
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_batchnorm(x, 96, 1, name='Inception_A_block' + '_conv1')
    branch_1 = conv2d_batchnorm(x, 48, 1, name='Inception_A_block' + '_conv2')
    branch_1 = conv2d_batchnorm(branch_1, 64, 5, name='Inception_A_block' + '_conv3')
    branch_2 = conv2d_batchnorm(x, 64, 1, name='Inception_A_block' + '_conv4')
    branch_2 = conv2d_batchnorm(branch_2, 96, 3, name='Inception_A_block' + '_conv5')
    branch_2 = conv2d_batchnorm(branch_2, 96, 3, name='Inception_A_block' + '_conv6')
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_batchnorm(branch_pool, 64, 1, name='Inception_A_block' + '_conv7')
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_batchnorm(x, 384, 3, strides=2, padding='valid', name='Reduction_A_block' + '_conv1')
    branch_1 = conv2d_batchnorm(x, 256, 1, name='Reduction_A_block' + '_conv2')
    branch_1 = conv2d_batchnorm(branch_1, 256, 3, name='Reduction_A_block' + '_conv3')
    branch_1 = conv2d_batchnorm(branch_1, 384, 3, strides=2, padding='valid', name='Reduction_A_block' + '_conv4')
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)

    return x


def classifier_layers(x, input_shape, trainable=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 4

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_batchnorm_timedist(x, 256, 1, name='Reduction_B_block' + '_conv1')
    branch_0 = conv2d_batchnorm_timedist(branch_0, 384, 3, strides=2, padding='valid', name='Reduction_B_block' + '_conv2')
    branch_1 = conv2d_batchnorm_timedist(x, 256, 1, name='Reduction_B_block' + '_conv3')
    branch_1 = conv2d_batchnorm_timedist(branch_1, 288, 3, strides=2, padding='valid', name='Reduction_B_block' + '_conv4')
    branch_2 = conv2d_batchnorm_timedist(x, 256, 1, name='Reduction_B_block' + '_conv5')
    branch_2 = conv2d_batchnorm_timedist(branch_2, 288, 3, name='Reduction_B_block' + '_conv6')
    branch_2 = conv2d_batchnorm_timedist(branch_2, 320, 3, strides=2, padding='valid', name='Reduction_B_block' + '_conv7')
    branch_pool = TimeDistributed(MaxPooling2D(3, strides=2, padding='valid'))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block_td(x, scale=0.2, block_type='block8', block_idx=block_idx)
    x = inception_resnet_block_td(x, scale=1., activation=None, block_type='block8', block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_batchnorm_timedist(x, 1536, 1, name='conv_7b')

    TimeDistributed(GlobalAveragePooling2D(), name='avg_pool')(x)
    return x

