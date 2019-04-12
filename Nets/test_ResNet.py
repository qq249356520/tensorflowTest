#-*- coding:utf-8 -*-

import tensorflow as tf

from __future__ import absolute_import #将新版本的特性引进当前版本，也就是说我们可以在当前版本使用新版本的一些特性

__BATCH_NORM_DECAY = 0.997
__BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16, )
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

#Convenience functions for building the ResNet model
def batch_norm(inputs, training, data_format):
    #Performs a batch normalization using a standard set of parameters
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels first' else 3,
        momentum=__BATCH_NORM_DECAY, epsilon=__BATCH_NORM_EPSILON,center=True,
        scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2 #除以并向下取整
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

#ResNet block definitions
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """
    A single block for ResNet v1, without a bottleneck
    Conv + BN + ReLU

    :param inputs: A tensor of size[batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format
    :param filters:
    :param training:
    :param projection_shortcut: The function to use for projection shortcuts
    (typically a 1*1 convolution when downsampling the input)
    :param strides:
    :param data_format:
    :return:
    """
    shortcur = inputs

    if projection_shortcut is not None:
        shortcur = projection_shortcut(inputs)
        shortcur = batch_norm(shortcur, training, data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcur
    inputs = tf.nn.relu(inputs)

    return inputs

def _building_block_v2(inputs, filters, training, projection_shortcut,
                       strides, data_format):
    #resnet v2
    shortcur = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcur = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcur

def _bottleneck_block_v1(inputs, filters, training, projection_shortcur,
                         strides, data_format):
    #v1 with a bottleneck
    shortcut = inputs

    if projection_shortcur is not None:
        shortcut = projection_shortcur(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                       strides, data_format):
    #resnet v2
    shortcur = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    if projection_shortcut is not None:
        shortcur = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcur

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name,
                data_format):
    """
    Create one layer of blocks for the ResNet model.

    :param inputs:
    :param filters:
    :param bottleneck:
    :param block_fn:
    :param blocks:
    :param strides:
    :param training:
    :param name:
    :param data_format:
    :return:
    """

    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcur(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)


    #Only the first block per block_layer uses projection_shortcur and strides
    inputs = block_fn(inputs, filters, training, projection_shortcur, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)

class Model(object):
    def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
                 kernel_size,
                 conv_stride, first_pool_size, first_pool_stride,
                 block_sizes, block_strides,
                 resnet_version=DEFAULT_VERSION, data_format=None,
                 dtype=DEFAULT_DTYPE):
        """
        Creates a model for classifying an image.

        :param resnet_size:A single integer for the size of the ResNet model
        :param bottleneck: regular blocks or bottleneck blocks
        :param num_classes:
        :param num_filters:
        :param kernel_size:
        :param conv_stride:
        :param first_pool_size:
        :param first_pool_stride:
        :param block_sizes:
        :param block_strides:
        :param resnet_version:
        :param data_format:
        :param dtype:
        """

        self.res_size = resnet_size

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise  ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_class = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_stride = block_strides
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                             *args, **kwargs):
        """
        Creates variables in fp32, then casts to fp16 if necessary.

        Args:
            param getter:The underlying variable getter,that has the same signature as
            tf.get_variable and returns a variable
            param name:
            param shape:
            param dtype: The dtype of the variable to get.Note that if this is a low
                precision dtype, the variable will be created as a tf.float32 variable,
                then cast to the appropriate dtype
            param args:
            param kwargs:
        :return:A variable which is cast to fp16 if necessary
        """

        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """
        Return a variable scope that the model should be created under.

        :return:A variable scope for the model
        """
        return tf.variable_scope('resnet_model', custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """
        Add operations to classify a batch of input images.

        Args:
            param inputs: A tensor representing a batch of input images
            param training: Aboolean.
        :return:A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters,kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_stride[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)

            if self.pre_activation:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            inputs = tf.layers.dense(inputs=inputs, units=self.num_class)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs



