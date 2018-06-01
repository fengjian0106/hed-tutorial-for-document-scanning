#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import const
from hed_net import *

from tensorflow import flags
flags.DEFINE_string('checkpoint_dir', './checkpoint', 
                    'Checkpoint directory.')
FLAGS = flags.FLAGS


if __name__ == "__main__":
    hed_graph_without_weights_file_name = 'hed_graph_without_weights.pb'
    hed_graph_without_weights_file_path = os.path.join(FLAGS.checkpoint_dir, hed_graph_without_weights_file_name)
    hed_graph_file_path = os.path.join(FLAGS.checkpoint_dir, 'hed_graph.pb')
    hed_tflite_model_file_path = os.path.join(FLAGS.checkpoint_dir, 'hed_lite_model.tflite')

    batch_size = 1
    image_float = tf.placeholder(tf.float32, shape=(batch_size, const.image_height, const.image_width, 3), name='hed_input')
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
    print('###1 image_float shape is: {}, name is: {}'.format(image_float.get_shape(), image_float.name))
    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = mobilenet_v2_style_hed(image_float, batch_size, is_training_placeholder)
    print('###2 dsn_fuse shape is: {}, name is: {}'.format(dsn_fuse.get_shape(), dsn_fuse.name))

    # Saver
    hed_weights  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    saver = tf.train.Saver(hed_weights)

    global_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_init)

        latest_ck_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_ck_file:
            print('restore from latest checkpoint file : {}'.format(latest_ck_file))
            saver.restore(sess, latest_ck_file)
        else:
            print('no checkpoint file to restore, exit()')
            exit()


        # C++ 代码中需要用到下面这三个 node 的 name 信息
        '''
        Input Node is:
           Tensor("hed_input:0", shape=(1, 256, 256, 3), dtype=float32)
           Tensor("is_training:0", dtype=bool)
        Output Node is:
           Tensor("hed/dsn_fuse/conv2d/BiasAdd:0", shape=(1, 256, 256, 1), dtype=float32)
        '''
        print('#######################################################')
        print('#######################################################')
        print('Input Node is:')
        print('   %s' % image_float)
        print('   %s' % is_training_placeholder)
        print('Output Node is:')
        print('   %s' % dsn_fuse)
        print('#######################################################')
        print('#######################################################')


        ################################################
        ################################################
        # Write input graph pb file
        tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.checkpoint_dir, hed_graph_without_weights_file_name)

        # We save out the graph to disk, and then call the const conversion routine.
        input_saver_def_path = ''
        input_binary = False
        input_checkpoint_path = latest_ck_file
        output_node_names = 'hed/dsn_fuse/conv2d/BiasAdd' ## !! 不是 'hed/dsn_fuse/conv2d/BiasAdd:0'
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        clear_devices = False
        # TensorFlow自带的这个freeze_graph函数，文档解释的不清楚，TODO
        freeze_graph.freeze_graph(hed_graph_without_weights_file_path, input_saver_def_path,
                                input_binary, input_checkpoint_path,
                                output_node_names, restore_op_name,
                                filename_tensor_name, hed_graph_file_path,
                                clear_devices, '')
        ################################################
        print('freeze to pb model finished')



        '''
        ## https://github.com/tensorflow/tensorflow/issues/17501, lite 版目前不支持 TransposeConv
        ################################################
        ################################################
        ################################################
        ## 遇到了这样一个 bug，https://github.com/tensorflow/tensorflow/issues/15410 Calling tf.contrib.lite.toco_convert results in global name 'tempfile' is not defined error
        ## TensorFlow 主干代码目前还未修复，先用下面这个临时方案处理一下
        ## manually put back imported modules
        import tempfile
        import subprocess
        tf.contrib.lite.tempfile = tempfile
        tf.contrib.lite.subprocess = subprocess
        ################################################

        print('tf.contrib.lite is: %r' % (tf.contrib.lite))
        # https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/lite/toco_convert
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite
        # 
        # tf.contrib.lite.toco_convert(
        #     input_data,
        #     input_tensors,
        #     output_tensors,
        #     inference_type=FLOAT,
        #     input_format=TENSORFLOW_GRAPHDEF,
        #     output_format=TFLITE,
        #     quantized_input_stats=None,
        #     drop_control_dependency=True
        # )
        tflite_model = tf.contrib.lite.toco_convert( \
                                    sess.graph_def, 
                                    [image_float,
                                     is_training_placeholder], 
                                    [dsn_fuse])
        open(hed_tflite_model_file_path, 'wb').write(tflite_model)
        print('tf.contrib.lite.toco_convert finished')
        '''
