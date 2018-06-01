#!/usr/bin/python
#coding=utf-8

'''
#############################################
这份 mobilenet.py 位于 unet-tutorial 目录里，
目前这个是最新版本，添加了 v2 version，
其他的工程可以直接使用这份代码或参考这里的实现。
#############################################
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import const

import tensorflow as tf


'''
alpha 对应论文的参数 Width Multiplier: Thinner Models

论文中还有另外一个超参数 ==> Resolution Multiplier: Reduced Representation
这个其实就是影响 input image 的 size

The second hyper-parameter to reduce the computational
cost of a neural network is a resolution multiplier ρ. 
We apply this to the input image and the internal representation of
every layer is subsequently reduced by the same multiplier.
In practice we implicitly set ρ by setting the input resolution
'''
def mobilenet_v1(inputs, alpha, is_training):
    '''
    https://arxiv.org/pdf/1704.04861v1.pdf  
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

    reference code
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
    https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/applications/mobilenet.py
    '''
    assert const.use_batch_norm == True

    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('alpha can be one of'
                         '`0.25`, `0.50`, `0.75` or `1.0` only.')

    filter_initializer = tf.contrib.layers.xavier_initializer()

    def _conv2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            outputs = tf.layers.conv2d(inputs,
                                    filters, 
                                    kernel_size, 
                                    strides=(stride, stride),
                                    padding='same', 
                                    activation=None,
                                    use_bias=False, 
                                    kernel_initializer=filter_initializer)
            '''
            https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Solutions.ipynb
            https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            '''
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = tf.nn.relu(outputs)
        return outputs
        
    def _depthwise_conv2d(inputs, 
                          pointwise_conv_filters, 
                          depthwise_conv_kernel_size,
                          stride,
                          scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('depthwise_conv'):
                '''
                tf.layers Module 里面有一个 tf.layers.separable_conv2d 函数，
                但是它的内部调用流程是 depthwise convolution --> pointwise convolution --> activation func，
                而 MobileNet V1 风格的卷积层的内部调用流程应该是
                depthwise conv --> batch norm --> relu --> pointwise conv --> batch norm --> relu，
                所以需要用其他的手段组装出想要的调用流程，
                一种办法是使用 tf.nn.depthwise_conv2d，但是这个 API 比较底层，代码写起来很笨重。
                后来找到了另外一种可行的办法，借助 tf.contrib.layers.separable_conv2d 函数，
                tf.contrib.layers.separable_conv2d 的第二个参数 num_outputs 如果设置为 None，
                则只会调用内部的 depthwise conv2d 部分，而不执行 pointwise conv2d 部分。
                这样就可以组装出 MobileNet V1 需要的 layer 结构了。


                TensorFlow 提供了四种 API，都命名为 separable_conv2d，但是又存在各种细微的差别，
                有兴趣的读者可以自行阅读相关文档
                tf.contrib.layers.separable_conv2d [Aliases tf.contrib.layers.separable_convolution2d]
                VS
                tf.keras.backend.separable_conv2d
                VS
                tf.layers.separable_conv2d
                VS
                tf.nn.separable_conv2d
                '''
                outputs = tf.contrib.layers.separable_conv2d(
                            inputs,
                            None, # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
                            depthwise_conv_kernel_size,
                            depth_multiplier=1,
                            stride=(stride, stride), # stride is just for tf.layers.separable_conv2d
                            padding='SAME',
                            activation_fn=None,
                            weights_initializer=filter_initializer,
                            biases_initializer=None) 
                
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('pointwise_conv'):
                pointwise_conv_filters = int(pointwise_conv_filters * alpha)
                outputs = tf.layers.conv2d(outputs,
                                        pointwise_conv_filters, 
                                        (1, 1), 
                                        padding='same', 
                                        activation=None,
                                        use_bias=False, 
                                        kernel_initializer=filter_initializer)

                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

        return outputs
        
    def _avg_pool2d(inputs, scope=''):
        inputs_shape = inputs.get_shape().as_list()
        assert len(inputs_shape) == 4

        pool_height = inputs_shape[1]
        pool_width = inputs_shape[2]
        
        with tf.variable_scope(scope):
            outputs = tf.layers.average_pooling2d(inputs,
                                            [pool_height, pool_width],
                                            strides=(1, 1),
                                            padding='valid')
        
        return outputs

    '''
    执行分类任务的网络结构，通常还可以作为实现其他任务的网络结构的 base architecture，
    为了方便代码复用，这里只需要实现出卷积层构成的主体部分，
    外部调用者根据各自的需求使用这里返回的 output 和 end_points。
    比如对于分类任务，按照如下方式使用这个函数
    
    image_height = 224
    image_width = 224
    image_channels = 3
    
    x = tf.placeholder(tf.float32, [None, image_height, image_width, image_channels])
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    output, net = mobilenet_v1(x, 1.0, is_training)
    print('output shape is: %r' % (output.get_shape().as_list()))
    
    output = tf.layers.flatten(output)
    output = tf.layers.dense(output,
                        units=1024, # 1024 class
                        activation=None,
                        use_bias=True,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    print('output shape is: %r' % (output.get_shape().as_list()))
    '''

    with tf.variable_scope('mobilenet_v1', 'mobilenet_v1', [inputs]):
        end_points = {}
        net = inputs 

        net = _conv2d(net, 32, [3, 3], stride=2, scope='block0')
        # print('\r ++++ block0 shape: %s' % (net.get_shape().as_list()))
        end_points['block0'] = net
        net = _depthwise_conv2d(net, 64, [3, 3], stride=1, scope='block1')
        end_points['block1'] = net

        net = _depthwise_conv2d(net, 128, [3, 3], stride=2, scope='block2')
        end_points['block2'] = net
        net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block3')
        end_points['block3'] = net

        net = _depthwise_conv2d(net, 256, [3, 3], stride=2, scope='block4')
        end_points['block4'] = net
        net = _depthwise_conv2d(net, 256, [3, 3], stride=1, scope='block5')
        end_points['block5'] = net

        net = _depthwise_conv2d(net, 512, [3, 3], stride=2, scope='block6')
        end_points['block6'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block7')
        end_points['block7'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block8')
        end_points['block8'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block9')
        end_points['block9'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block10')
        end_points['block10'] = net
        net = _depthwise_conv2d(net, 512, [3, 3], stride=1, scope='block11')
        end_points['block11'] = net

        net = _depthwise_conv2d(net, 1024, [3, 3], stride=2, scope='block12')
        end_points['block12'] = net
        net = _depthwise_conv2d(net, 1024, [3, 3], stride=1, scope='block13')
        end_points['block13'] = net

        output = _avg_pool2d(net, scope='output')

    return output, end_points




def mobilenet_v2_func_blocks(is_training):
    assert const.use_batch_norm == True

    filter_initializer = tf.contrib.layers.xavier_initializer()
    activation_func = tf.nn.relu6

    def conv2d(inputs, filters, kernel_size, stride, scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('conv2d'):
                outputs = tf.layers.conv2d(inputs,
                                        filters, 
                                        kernel_size, 
                                        strides=(stride, stride),
                                        padding='same', 
                                        activation=None,
                                        use_bias=False, 
                                        kernel_initializer=filter_initializer)
                '''
                https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Solutions.ipynb
                https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                '''
                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)
            return outputs

    def _1x1_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs,
                                    filters, 
                                    kernel_size, 
                                    strides=(stride, stride),
                                    padding='same', 
                                    activation=None,
                                    use_bias=False, 
                                    kernel_initializer=filter_initializer)
            
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            # no activation_func
        return outputs

    def expansion_conv2d(inputs, expansion, stride):
        input_shape = inputs.get_shape().as_list()
        assert len(input_shape) == 4
        filters = input_shape[3] * expansion

        kernel_size = [1, 1]
        with tf.variable_scope('expansion_1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs,
                                    filters, 
                                    kernel_size, 
                                    strides=(stride, stride),
                                    padding='same', 
                                    activation=None,
                                    use_bias=False, 
                                    kernel_initializer=filter_initializer)

            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = activation_func(outputs)
        return outputs

    def projection_conv2d(inputs, filters, stride):
        kernel_size = [1, 1]
        with tf.variable_scope('projection_1x1_conv2d'):
            outputs = tf.layers.conv2d(inputs,
                                    filters, 
                                    kernel_size, 
                                    strides=(stride, stride),
                                    padding='same', 
                                    activation=None,
                                    use_bias=False, 
                                    kernel_initializer=filter_initializer)

            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            # no activation_func
        return outputs

    def depthwise_conv2d(inputs, 
                        depthwise_conv_kernel_size,
                        stride):
        with tf.variable_scope('depthwise_conv2d'):
            outputs = tf.contrib.layers.separable_conv2d(
                        inputs,
                        None, # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
                        depthwise_conv_kernel_size,
                        depth_multiplier=1,
                        stride=(stride, stride),
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=filter_initializer,
                        biases_initializer=None) 

            outputs = tf.layers.batch_normalization(outputs, training=is_training)
            outputs = activation_func(outputs)

        return outputs
        
    def avg_pool2d(inputs, scope=''):
        inputs_shape = inputs.get_shape().as_list()
        assert len(inputs_shape) == 4

        pool_height = inputs_shape[1]
        pool_width = inputs_shape[2]
        
        with tf.variable_scope(scope):
            outputs = tf.layers.average_pooling2d(inputs,
                                            [pool_height, pool_width],
                                            strides=(1, 1),
                                            padding='valid')
        
        return outputs

    def inverted_residual_block(inputs, 
                            filters, 
                            stride, 
                            expansion=6, 
                            scope=''):
        assert stride == 1 or stride == 2

        depthwise_conv_kernel_size = [3, 3]
        pointwise_conv_filters = filters
        
        with tf.variable_scope(scope):
            net = inputs
            net = expansion_conv2d(net, expansion, stride=1)
            net = depthwise_conv2d(net, depthwise_conv_kernel_size, stride=stride)
            net = projection_conv2d(net, pointwise_conv_filters, stride=1)

            if stride == 1:
                # print('----------------- test, net.get_shape().as_list()[3] = %r' % net.get_shape().as_list()[3])
                # print('----------------- test, inputs.get_shape().as_list()[3] = %r' % inputs.get_shape().as_list()[3])
                # 如果 net.get_shape().as_list()[3] != inputs.get_shape().as_list()[3]
                # 借助一个 1x1 的卷积让他们的 channels 相等，然后再相加
                if net.get_shape().as_list()[3] != inputs.get_shape().as_list()[3]:
                    inputs = _1x1_conv2d(inputs, net.get_shape().as_list()[3], stride=1)

                net = net + inputs
                return net
            else:
                # stride == 2
                return net

    func_blocks = {}
    func_blocks['conv2d'] = conv2d
    func_blocks['inverted_residual_block'] = inverted_residual_block
    func_blocks['avg_pool2d'] = avg_pool2d
    func_blocks['filter_initializer'] = filter_initializer
    func_blocks['activation_func'] = activation_func

    return func_blocks


def mobilenet_v2(inputs, is_training):
    assert const.use_batch_norm == True

    func_blocks = mobilenet_v2_func_blocks(is_training)
    _conv2d = func_blocks['conv2d'] 
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']

    with tf.variable_scope('mobilenet_v2', 'mobilenet_v2', [inputs]):
        end_points = {}
        net = inputs 
    
        net = _conv2d(net, 32, [3, 3], stride=2, scope='block0_0') # size/2
        end_points['block0'] = net
        print('!! debug block0, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 16, stride=1, expansion=1, scope='block1_0')
        end_points['block1'] = net
        print('!! debug block1, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 24, stride=2, scope='block2_0') # size/4
        net = _inverted_residual_block(net, 24, stride=1, scope='block2_1')
        end_points['block2'] = net
        print('!! debug block2, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 32, stride=2, scope='block3_0') # size/8
        net = _inverted_residual_block(net, 32, stride=1, scope='block3_1') 
        net = _inverted_residual_block(net, 32, stride=1, scope='block3_2')
        end_points['block3'] = net
        print('!! debug block3, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 64, stride=2, scope='block4_0') # size/16
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_1') 
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_2') 
        net = _inverted_residual_block(net, 64, stride=1, scope='block4_3') 
        end_points['block4'] = net
        print('!! debug block4, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 96, stride=1, scope='block5_0') 
        net = _inverted_residual_block(net, 96, stride=1, scope='block5_1')
        net = _inverted_residual_block(net, 96, stride=1, scope='block5_2')
        end_points['block5'] = net
        print('!! debug block5, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 160, stride=2, scope='block6_0') # size/32
        net = _inverted_residual_block(net, 160, stride=1, scope='block6_1') 
        net = _inverted_residual_block(net, 160, stride=1, scope='block6_2') 
        end_points['block6'] = net
        print('!! debug block6, net shape is: {}'.format(net.get_shape()))

        net = _inverted_residual_block(net, 320, stride=1, scope='block7_0')
        end_points['block7'] = net
        print('!! debug block7, net shape is: {}'.format(net.get_shape()))

        net = _conv2d(net, 1280, [1, 1], stride=1, scope='block8_0') 
        end_points['block8'] = net
        print('!! debug block8, net shape is: {}'.format(net.get_shape()))

        output = _avg_pool2d(net, scope='output')
        print('!! debug after avg_pool, net shape is: {}'.format(output.get_shape()))

    return output, end_points