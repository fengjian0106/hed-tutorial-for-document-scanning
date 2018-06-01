#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import const

#######################################################################
#######################################################################
#######################################################################
# import matplotlib.pyplot as plt
# virtualenv 环境里直接 import matplotlib.pyplot as plt 会报错，解决办法如下
# https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#######################################################################
#######################################################################
#######################################################################



## VGG mean pixel
R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

# Subtracts the given means from each image channel.
def mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input Tensor image must be of shape [height, width, 3]')

    num_channels = image.get_shape().as_list()[-1]
    if num_channels != 3:
        raise ValueError('Input Tensor image must have 3 channels')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    rgb_channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        # print('____________, i={}, aftrer split, channels shape is: {}'.format(i, rgb_channels[i].get_shape()))
        rgb_channels[i] -= means[i]

    return tf.concat(rgb_channels, 2)




def plot_and_save_image(input_image_path, input_annotation,
                        dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5,
                        dir, suffix=''):
    if not os.path.exists(dir):
        os.makedirs(dir) 

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharey=True, figsize=(12, 12))
    fig.tight_layout()

    ax1.set_title('input image')
    input_image = cv2.imread(input_image_path)
    b, g, r  = cv2.split(input_image)
    input_image = cv2.merge((r, g, b))
    input_image = cv2.resize(input_image, (const.image_width, const.image_height), interpolation = cv2.INTER_CUBIC)
    ax1.imshow(input_image)
    ax1.axis('off')

    ax2.set_title('ground truth')
    ax2.imshow(np.dstack((input_annotation,)*3))
    ax2.axis('off')

    '''
    dsn_fuse 中的值，并不是像 label image 那样落在 (0.0, 1.0) 这个区间范围内的，
    用 threshold 处理一下，就可以转换成对应 image 的矩阵，让像素值落在正常取值区间内。

    像下面这样不做转换，直接绘制其实也能看到边缘线
    ax3.set_title('dsn_fuse')
    ax3.imshow(np.dstack((dsn_fuse,)*3))
    ax3.axis('off')
    '''
    threshold = 0.0
    dsn_fuse_image = np.where(dsn_fuse > threshold, 1.0, 0.0)
    ax3.set_title('dsn_fuse')
    ax3.imshow(np.dstack((dsn_fuse_image,)*3))
    ax3.axis('off')
    
    '''
    dsn1 -- dsn5，只中间过程的 Tensor，矩阵元素的值不在 (0.0, 1.0) 这个区间范围内是正常的。
    如果也想用 threshold 做一个处理，需要看一下每一个 dsn 矩阵里元素值的大致分布情况，然后挑选一个 threshold。
    后面的几行代码，是没有做 threshold 处理的。
    '''
    ax4.set_title('dsn1')
    ax4.imshow(np.dstack((dsn1,)*3))
    ax4.axis('off')

    ax5.set_title('dsn2')
    ax5.imshow(np.dstack((dsn2,)*3))
    ax5.axis('off')
    
    ax6.set_title('dsn3')
    ax6.imshow(np.dstack((dsn3,)*3))
    ax6.axis('off')

    ax7.set_title('dsn4')
    ax7.imshow(np.dstack((dsn4,)*3))
    ax7.axis('off')
    
    ax8.set_title('dsn5')
    ax8.imshow(np.dstack((dsn5,)*3))
    ax8.axis('off')

    image_path = os.path.join(dir, 'test-{}.png'.format(suffix))
    fig.savefig(image_path)
    plt.close(fig)  
