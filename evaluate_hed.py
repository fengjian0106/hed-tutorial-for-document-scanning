#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import time
import tensorflow as tf

import const
from util import *
from hed_net import *

from tensorflow import flags
flags.DEFINE_string('image', './test_image/test15.jpg', 
                    'Image path to run hed, must be jpg image.')
flags.DEFINE_string('checkpoint_dir', './checkpoint', 
                    'Checkpoint directory.')
flags.DEFINE_string('output_dir', './test_image', 
                    'Output directory.')
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.image):
    print('image {} not exists, please retry' % FLAGS.image)
    exit()

if __name__ == "__main__":
    batch_size = 1
    image_path_placeholder = tf.placeholder(tf.string)
    is_training_placeholder = tf.placeholder(tf.bool)

    feed_dict_to_use = {image_path_placeholder: FLAGS.image,
                        is_training_placeholder: False}


    image_tensor = tf.read_file(image_path_placeholder)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.image.resize_images(image_tensor, [const.image_height, const.image_width])

    image_float = tf.to_float(image_tensor)

    if const.use_batch_norm == True:
        image_float = image_float / 255.0
    else:
        # for VGG style HED net
        image_float = mean_image_subtraction(image_float, [R_MEAN, G_MEAN, B_MEAN])
    image_float = tf.expand_dims(image_float, axis=0)

    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = mobilenet_v2_style_hed(image_float, batch_size, is_training_placeholder)

    global_init = tf.global_variables_initializer()

    # Saver
    hed_weights  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    saver = tf.train.Saver(hed_weights)

    with tf.Session() as sess:
        sess.run(global_init)

        latest_ck_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_ck_file:
            print('restore from latest checkpoint file : {}'.format(latest_ck_file))
            saver.restore(sess, latest_ck_file)
        else:
            print('no checkpoint file to restore, exit()')
            exit()

        _dsn_fuse, \
        _dsn1, \
        _dsn2, \
        _dsn3, \
        _dsn4, \
        _dsn5 = sess.run([dsn_fuse,
                        dsn1, dsn2,
                        dsn3, dsn4,
                        dsn5],
                        feed_dict=feed_dict_to_use)

        # print('%r' % _dsn_fuse)

        '''
        HED 网络输出的 Tensor 中的像素值，并不是像 label image 那样落在 (0.0, 1.0) 这个区间范围内的。
        用 threshold 处理一下，就可以转换成对应 image 的矩阵，让像素值落在正常取值区间内
        '''
        threshold = 0.0
        dsn_fuse_image = np.where(_dsn_fuse[0] > threshold, 255, 0)
        dsn_fuse_image_path = os.path.join(FLAGS.output_dir, 'dsn_fuse.png')
        cv2.imwrite(dsn_fuse_image_path, dsn_fuse_image)
