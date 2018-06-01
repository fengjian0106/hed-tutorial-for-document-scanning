#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np

import tensorflow as tf
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion('1.6'), 'Please use TensorFlow version 1.6 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

import const
from util import *
from input_pipeline import *
from hed_net import *

from tensorflow import flags
flags.DEFINE_string('dataset_root_dir', '', 
                    'Root directory to put all the training data.')
flags.DEFINE_string('csv_path', '', 
                    'CSV file path.')
flags.DEFINE_string('checkpoint_dir', './checkpoint', 
                    'Checkpoint directory.')
flags.DEFINE_string('debug_image_dir', './debug_output_image', 
                    'Debug output image directory.')
flags.DEFINE_string('log_dir', './log', 
                    'Log directory for tensorflow.')
flags.DEFINE_integer('batch_size', 4, 
                     'Batch size, default 4.')
flags.DEFINE_integer('iterations', 90000000, 
                     'Number of iterations, default 90000000.')
flags.DEFINE_integer('display_step', 20, 
                     'Number of iterations between optimizer print info and save test image, default 20.')
flags.DEFINE_float('learning_rate', 0.0005, 
                   'Learning rate, default 0.0005.')
flags.DEFINE_boolean('restore_checkpoint', True, 
                     'If true, restore from latest checkpoint, default True.')
flags.DEFINE_boolean('just_set_batch_size_to_one', False, 
                     'If true, just set batch size to one and exit(in order to call python freeze_model.py), default False.')
FLAGS = flags.FLAGS

if FLAGS.dataset_root_dir == '':
    print('must set dataset_root_dir')
    exit()
if FLAGS.csv_path == '':
    print('must set csv_path')
    exit()

if FLAGS.just_set_batch_size_to_one:
    FLAGS.batch_size = 1

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.debug_image_dir):
    os.makedirs(FLAGS.debug_image_dir)
hed_ckpt_file_path = os.path.join(FLAGS.checkpoint_dir, 'hed.ckpt')
print('###########################################')
print('###########################################')
print('dataset_root_dir is: {}'.format(FLAGS.dataset_root_dir))
print('os.path.join(FLAGS.dataset_root_dir, \'\') is: {}'.format(os.path.join(FLAGS.dataset_root_dir, '')))
print('csv_path is: {}'.format(FLAGS.csv_path))
print('checkpoint_dir is: {}'.format(FLAGS.checkpoint_dir))
print('debug_image_dir is: {}'.format(FLAGS.debug_image_dir))
print('###########################################')
print('###########################################')




if __name__ == "__main__":
    #命令行传入的路径参数，不带最后的'/'，这里要把'/'补全，然后传入给fix_size_image_pipeline
    dataset_root_dir_string = os.path.join(FLAGS.dataset_root_dir, '') 

    '''
    严格来说，在机器学习任务，应该区分训练集和验证集。
    但是在这份代码中，因为训练样本都是合成出来的，所以就没有区分验证集了，
    而是直接通过肉眼观察 FLAGS.debug_image_dir 目录里输出的 debug image 来判断是否可以结束训练，
    然后直接放到真实的使用环境里判断模型的实际使用效果。

    另外，这个任务里面，评估训练效果的时候，也没有计算 lable 和 output 之间的 IOU 值，原因如下：
    我用执行 Semantic Segmentation 任务的 UNet 网络也尝试过做这个边缘检测任务，
    在这个合成的训练样本上，UNet 的 IOU 值是远好于 HED 网络的，
    但是在真实使用的场景里，UNet 的效果则不如 HED 了，
    HED 检测到的边缘线是有 "过剩" 的部分的，比如边缘线比样本中的边缘线更粗、同时还会检测到一些干扰边缘线，
    这些 "过剩" 的部分，可以借助后续流程里的找点算法逐层过滤掉， 
    而 UNet 的效果就正好相反了，边缘线有些时候会遇到 "缺失"，而且可能会缺失掉关键的部分，比如矩形区域的拐角处，
    这种关键部位的 "缺失"，在后续的找点算法里就有点无能为力。 
    '''
    input_queue_for_train = tf.train.string_input_producer([FLAGS.csv_path])
    image_tensor, annotation_tensor, \
        image_path_tensor = fix_size_image_pipeline(dataset_root_dir_string, 
                                            input_queue_for_train, 
                                            FLAGS.batch_size)

    '''
    # 常规情况下的代码，这里还应该有一个读取 verify 数据的 pipeline
    input_queue_for_verify = tf.train.string_input_producer([FLAGS.validation_data_file_path])
    image_tensor_for_verify, annotation_tensor_for_verify, \
        image_path_tensor_for_verify = fix_size_image_pipeline(dataset_root_dir_string, 
                                                input_queue_for_verify, 
                                                FLAGS.batch_size)
    '''

    



    is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
    feed_dict_to_use = {is_training_placeholder: True}

    print('image_tensor shape is: {}'.format(image_tensor.get_shape()))
    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = mobilenet_v2_style_hed(image_tensor, 
                                                            FLAGS.batch_size, 
                                                            is_training_placeholder)
    print('dsn_fuse shape is: {}'.format(dsn_fuse.get_shape()))


    cost = class_balanced_sigmoid_cross_entropy(dsn_fuse, annotation_tensor) 
    if const.use_kernel_regularizer:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        '''
        for reg_loss in reg_losses:
            print('reg_loss shape is: {}'.format(reg_loss.get_shape()))
        '''
        reg_constant = 0.0001
        cost = cost + reg_constant * sum(reg_losses)

    print('cost shape is: {}'.format(cost.get_shape()))
    cost_reduce_mean = tf.reduce_mean(cost) # for tf.summary

    
    with tf.variable_scope("adam_vars"):
        if const.use_batch_norm == True:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
        
    global_init = tf.global_variables_initializer()

    # summary
    tf.summary.scalar('loss', cost_reduce_mean) 
    merged_summary_op = tf.summary.merge_all()
    summary_string_writer = tf.summary.FileWriter(FLAGS.log_dir)


    # saver
    hed_weights  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    all_variables_can_restore = hed_weights # 还可以加上其他的 var，整体就是个 [] 数组
    # print('===============================')
    # print('===============================')
    # print('===============================')
    # print('all_variables_can_restore are:')
    # for tensor_var in all_variables_can_restore:
    #     print('  %r' % (tensor_var))
    # print('===============================')
    # print('===============================')
    # print('===============================')
    ckpt_saver = tf.train.Saver(all_variables_can_restore)

    print('\n\n')
    print('############################################################')
    with tf.Session() as sess:
        sess.run(global_init)

        if FLAGS.restore_checkpoint:
            latest_ck_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if latest_ck_file:
                print('restore from latest checkpoint file : {}'.format(latest_ck_file))
                ckpt_saver.restore(sess, latest_ck_file)
            else:
                print('no checkpoint file to restore')
        else:
            print('no checkpoint file to restore')

        ##############
        if FLAGS.just_set_batch_size_to_one:
            ckpt_saver.save(sess, hed_ckpt_file_path, global_step=0)
            exit()
        ##############
        

        print('\nStart train...')
        print('batch_size is: {}'.format(FLAGS.batch_size))
        print('iterations is: {}'.format(FLAGS.iterations))
        print('display-step is: {}'.format(FLAGS.display_step))
        print('learning-rate is: {}'.format(FLAGS.learning_rate))
        if const.use_kernel_regularizer:
            print('++ use l2 regularizer')
        if const.use_batch_norm == True:
            print('++ use batch norm')
        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

        for step in range(FLAGS.iterations):
            feed_dict_to_use[is_training_placeholder] = True
            loss_mean, loss, summary_string = sess.run([cost_reduce_mean, cost, merged_summary_op],
                                            feed_dict=feed_dict_to_use)
            sess.run(train_step, feed_dict=feed_dict_to_use)
            
            summary_string_writer.add_summary(summary_string, step)
            

            if step % FLAGS.display_step == 0:
                ckpt_saver.save(sess, hed_ckpt_file_path, global_step=step)

                feed_dict_to_use[is_training_placeholder] = False

                _input_image_path, _input_annotation, \
                _loss_mean, \
                _dsn_fuse, \
                _dsn1, \
                _dsn2, \
                _dsn3, \
                _dsn4, \
                _dsn5 = sess.run([image_path_tensor, annotation_tensor,
                                    cost_reduce_mean,
                                    dsn_fuse,
                                    dsn1, dsn2,
                                    dsn3, dsn4,
                                    dsn5],
                                    feed_dict=feed_dict_to_use)
                print("Step: {}, Current Mean Loss: {}".format(step, loss_mean))

                plot_and_save_image(_input_image_path[0], _input_annotation[0],
                                    _dsn_fuse[0], _dsn1[0], _dsn2[0], _dsn3[0], _dsn4[0], _dsn5[0],
                                    FLAGS.debug_image_dir, suffix='{}'.format(step))

        ###########
        coord.request_stop()
        coord.join(threads)
        print("Train Finished!")
