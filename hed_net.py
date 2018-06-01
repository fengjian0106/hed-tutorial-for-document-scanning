#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import const
from mobilenet import *


def class_balanced_sigmoid_cross_entropy(logits, label):
    ## ref - https://github.com/ppwwyyxx/tensorpack/blob/master/examples/HED/hed.py
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        count_neg = tf.reduce_sum(1.0 - label) # 样本中0的数量
        count_pos = tf.reduce_sum(label) # 样本中1的数量(远小于count_neg)
        # print('debug, ==========================, count_pos is: {}'.format(count_pos))
        beta = count_neg / (count_neg + count_pos)  ## e.g.  60000 / (60000 + 800) = 0.9868

        pos_weight = beta / (1.0 - beta)  ## 0.9868 / (1.0 - 0.9868) = 0.9868 / 0.0132 = 74.75
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))

        # 如果样本中1的数量等于0，那就直接让 cost 为 0，因为 beta == 1 时， 除法 pos_weight = beta / (1.0 - beta) 的结果是无穷大
        zero = tf.equal(count_pos, 0.0)
        final_cost = tf.where(zero, 0.0, cost) 
    return final_cost



def mobilenet_v2_style_hed(inputs, batch_size, is_training):
    assert const.use_batch_norm == True
    assert const.use_kernel_regularizer == False

    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    else:
        weights_regularizer = None

    ####################################################
    func_blocks = mobilenet_v2_func_blocks(is_training)
    # print('============ func_blocks are: %r' % func_blocks)
    _conv2d = func_blocks['conv2d'] 
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']
    filter_initializer = func_blocks['filter_initializer']
    activation_func = func_blocks['activation_func']
    ####################################################

    def _dsn_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=False, 
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        ## no activation

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=True, ## use bias
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        ## no batch normalization
        ## no activation

        return outputs


    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        ## https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
        ## 从这个原版代码里看，是这样计算 kernel_size 的
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters, 
                                             kernel_size, 
                                             strides=(upsample_factor, upsample_factor), 
                                             padding='same', 
                                             activation=None, ## no activation
                                             use_bias=True, ## use bias
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)

        ## 概念上来说，deconv2d 已经是最后的输出 layer 了，只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        ## 所以不需要再使用 batch normalization 了

        return outputs


    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs
        

        ## mobilenet v2 as base net
        with tf.variable_scope('mobilenet_v2'):
            # 标准的 mobilenet v2 里面并没有这两层，
            # 这里是为了得到和 input image 相同 size 的 feature map 而增加的层
            net = _conv2d(net, 3, [3, 3], stride=1, scope='block0_0')
            net = _conv2d(net, 6, [3, 3], stride=1, scope='block0_1')

            dsn1 = net
            net = _conv2d(net, 12, [3, 3], stride=2, scope='block0_2') # size/2

            net = _inverted_residual_block(net, 6, stride=1, expansion=1, scope='block1_0')

            dsn2 = net
            net = _inverted_residual_block(net, 12, stride=2, scope='block2_0') # size/4
            net = _inverted_residual_block(net, 12, stride=1, scope='block2_1')

            dsn3 = net
            net = _inverted_residual_block(net, 24, stride=2, scope='block3_0') # size/8
            net = _inverted_residual_block(net, 24, stride=1, scope='block3_1') 
            net = _inverted_residual_block(net, 24, stride=1, scope='block3_2')

            dsn4 = net
            net = _inverted_residual_block(net, 48, stride=2, scope='block4_0') # size/16
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_1') 
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_2') 
            net = _inverted_residual_block(net, 48, stride=1, scope='block4_3') 

            net = _inverted_residual_block(net, 64, stride=1, scope='block5_0') 
            net = _inverted_residual_block(net, 64, stride=1, scope='block5_1')
            net = _inverted_residual_block(net, 64, stride=1, scope='block5_2')

            dsn5 = net


        ## dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            # print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            # print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor = 2)
            # print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            # print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor = 4)
            # print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            # print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor = 8)
            # print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            # print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor = 16)
            # print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))


        # dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            # print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            # print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5



def mobilenet_v1_style_hed(inputs, batch_size, is_training):
    assert const.use_batch_norm == True
    assert const.use_kernel_regularizer == False

    alpha = 1.0
    filter_initializer = tf.contrib.layers.xavier_initializer()

    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    else:
        weights_regularizer = None

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

    '''stride is just for tf.layers.separable_conv2d, means depthwise_conv_stride'''
    def _depthwise_conv2d(inputs, 
                          pointwise_conv_filters, 
                          depthwise_conv_kernel_size,
                          stride,
                          scope=''):
        with tf.variable_scope(scope):
            with tf.variable_scope('depthwise_conv'):
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
                '''
                !!!important!!! tf.contrib.layers.separable_conv2d already has a depthwise convolution and a pointwise convolution,
                but By passing num_outputs=None, separable_conv2d produces only a depthwise convolution layer 
                ref -- https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py 
                '''

            with tf.variable_scope('pointwise_conv'):
                pointwise_conv_filters = int(pointwise_conv_filters * alpha)
                outputs = tf.layers.conv2d(outputs,
                                        pointwise_conv_filters, ##!! here, pointwise_conv_filters * alpha
                                        (1, 1), 
                                        padding='same', 
                                        activation=None,
                                        use_bias=False, 
                                        kernel_initializer=filter_initializer)

                outputs = tf.layers.batch_normalization(outputs, training=is_training)
                outputs = tf.nn.relu(outputs)

        return outputs



    def _dsn_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=False, 
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        ## no activation

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=True, ## use bias
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        ## no batch normalization
        ## no activation

        return outputs


    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        ## https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
        ## 从这个原版代码里看，是这样计算 kernel_size 的
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters, 
                                             kernel_size, 
                                             strides=(upsample_factor, upsample_factor), 
                                             padding='same', 
                                             activation=None, ## no activation
                                             use_bias=True, ## use bias
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)

        ## 概念上来说，deconv2d 已经是最后的输出 layer 了，只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        ## 所以不需要再使用 batch normalization 了

        return outputs


    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        ## mobilenet v1 as base net
        with tf.variable_scope('mobilenet_v1'):
            # 标准的 mobilenet v1 里面并没有这两层，
            # 这里是为了得到和 input image 相同 size 的 feature map 而增加的层
            net = _conv2d(net, 6, [3, 3], stride=1, scope='extra_block0')
            net = _conv2d(net, 6, [3, 3], stride=1, scope='extra_block1')

            dsn1 = net
            net = _conv2d(net, 8, [3, 3], stride=2, scope='block0')
            # print('\r ++++ block0 shape: %s' % (net.get_shape().as_list()))
            end_points['block0'] = net
            net = _depthwise_conv2d(net, 16, [3, 3], stride=1, scope='block1')
            end_points['block1'] = net

            dsn2 = net
            net = _depthwise_conv2d(net, 32, [3, 3], stride=2, scope='block2')
            end_points['block2'] = net
            net = _depthwise_conv2d(net, 32, [3, 3], stride=1, scope='block3')
            end_points['block3'] = net

            dsn3 = net
            net = _depthwise_conv2d(net, 64, [3, 3], stride=2, scope='block4')
            end_points['block4'] = net
            net = _depthwise_conv2d(net, 64, [3, 3], stride=1, scope='block5')
            end_points['block5'] = net

            dsn4 = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=2, scope='block6')
            end_points['block6'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block7')
            end_points['block7'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block8')
            end_points['block8'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block9')
            end_points['block9'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block10')
            end_points['block10'] = net
            net = _depthwise_conv2d(net, 128, [3, 3], stride=1, scope='block11')
            end_points['block11'] = net

            dsn5 = net
            

        ## dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor = 2)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor = 4)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor = 8)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor = 16)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))


        # dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5



def vgg_style_hed(inputs, batch_size, is_training):
    filter_initializer = tf.contrib.layers.xavier_initializer()
    if const.use_kernel_regularizer:
        weights_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
    else:
        weights_regularizer = None

    def _vgg_conv2d(inputs, filters, kernel_size):
        use_bias = True
        if const.use_batch_norm:
            use_bias = False
            
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## call relu after batch normalization
                                   use_bias=use_bias,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)
        if const.use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        outputs = tf.nn.relu(outputs)
        return outputs

    def _max_pool2d(inputs):
        outputs = tf.layers.max_pooling2d(inputs, 
                                          [2, 2], 
                                          strides=(2, 2),
                                          padding='same')
        return outputs

    def _dsn_1x1_conv2d(inputs, filters):
        use_bias = True
        if const.use_batch_norm:
            use_bias = False

        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=use_bias, 
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        if const.use_batch_norm:
            outputs = tf.layers.batch_normalization(outputs, training=is_training)
        ## no activation

        return outputs

    def _output_1x1_conv2d(inputs, filters):
        kernel_size = [1, 1]
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size, 
                                   padding='same', 
                                   activation=None, ## no activation
                                   use_bias=True, ## use bias
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=weights_regularizer)

        ## no batch normalization
        ## no activation

        return outputs

    def _dsn_deconv2d_with_upsample_factor(inputs, filters, upsample_factor):
        ## https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
        ## 从这个原版代码里看，是这样计算 kernel_size 的
        kernel_size = [2 * upsample_factor, 2 * upsample_factor]
        outputs = tf.layers.conv2d_transpose(inputs,
                                             filters, 
                                             kernel_size, 
                                             strides=(upsample_factor, upsample_factor), 
                                             padding='same', 
                                             activation=None, ## no activation
                                             use_bias=True, ## use bias
                                             kernel_initializer=filter_initializer,
                                             kernel_regularizer=weights_regularizer)

        ## 概念上来说，deconv2d 已经是最后的输出 layer 了，只不过最后还有一步 1x1 的 conv2d 把 5 个 deconv2d 的输出再融合到一起
        ## 所以不需要再使用 batch normalization 了

        return outputs
        

    # ref https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
    with tf.variable_scope('hed', 'hed', [inputs]):
        end_points = {}
        net = inputs

        with tf.variable_scope('conv1'):
            net = _vgg_conv2d(net, 12, [3, 3])
            net = _vgg_conv2d(net, 12, [3, 3])
            dsn1 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv2'):
            net = _vgg_conv2d(net, 24, [3, 3])
            net = _vgg_conv2d(net, 24, [3, 3])
            dsn2 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv3'):
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            net = _vgg_conv2d(net, 48, [3, 3])
            dsn3 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv4'):
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            net = _vgg_conv2d(net, 96, [3, 3])
            dsn4 = net
            net = _max_pool2d(net)

        with tf.variable_scope('conv5'):
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            net = _vgg_conv2d(net, 192, [3, 3])
            dsn5 = net
            # net = _max_pool2d(net) # no need this pool layer

        ## dsn layers
        with tf.variable_scope('dsn1'):
            dsn1 = _dsn_1x1_conv2d(dsn1, 1)
            print('!! debug, dsn1 shape is: {}'.format(dsn1.get_shape()))
            ## no need deconv2d

        with tf.variable_scope('dsn2'):
            dsn2 = _dsn_1x1_conv2d(dsn2, 1)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))
            dsn2 = _dsn_deconv2d_with_upsample_factor(dsn2, 1, upsample_factor = 2)
            print('!! debug, dsn2 shape is: {}'.format(dsn2.get_shape()))

        with tf.variable_scope('dsn3'):
            dsn3 = _dsn_1x1_conv2d(dsn3, 1)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))
            dsn3 = _dsn_deconv2d_with_upsample_factor(dsn3, 1, upsample_factor = 4)
            print('!! debug, dsn3 shape is: {}'.format(dsn3.get_shape()))

        with tf.variable_scope('dsn4'):
            dsn4 = _dsn_1x1_conv2d(dsn4, 1)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))
            dsn4 = _dsn_deconv2d_with_upsample_factor(dsn4, 1, upsample_factor = 8)
            print('!! debug, dsn4 shape is: {}'.format(dsn4.get_shape()))

        with tf.variable_scope('dsn5'):
            dsn5 = _dsn_1x1_conv2d(dsn5, 1)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))
            dsn5 = _dsn_deconv2d_with_upsample_factor(dsn5, 1, upsample_factor = 16)
            print('!! debug, dsn5 shape is: {}'.format(dsn5.get_shape()))

        ##dsn fuse
        with tf.variable_scope('dsn_fuse'):
            dsn_fuse = tf.concat([dsn1, dsn2, dsn3, dsn4, dsn5], 3)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))
            dsn_fuse = _output_1x1_conv2d(dsn_fuse, 1)
            print('debug, dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5
