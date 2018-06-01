#!/usr/bin/python
#coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# http://code.activestate.com/recipes/65207-constants-in-python/
class _const:
    class ConstError(TypeError): pass
    def __setattr__(self,name,value):
        if self.__dict__.has_key(name):
            raise self.ConstError, "Can't rebind const(%s)"%name
        self.__dict__[name]=value
import sys
sys.modules[__name__]=_const()



import const
import os

const.image_height = 256
const.image_width = 256


'''
如果使用 mobilenet_v2_style_hed 或 mobilenet_v1_style_hed，
一定要设置 const.use_batch_norm = True，因为 MobileNet 本身就要求使用 batch norm。
L2 regularizer 可用可不用，目前我选择的是不使用。
'''
const.use_batch_norm = True 
const.use_kernel_regularizer = False
'''
如果使用 vgg_style_hed，分两种情况，
1、按照 VGG 的标准使用方法，是没有 batch norm 的，同时，input image 要减去一组 RGB channel 的平均值，
在 input_pipeline.py 里面有体现这一点。https://github.com/tensorflow/models/issues/517。
2、如果使用 batch norm，则和 mobilenet_v2_style_hed 类似，对于 input image 都统一进行归一化处理，缩放到 (0.0, 1.0) 这个数值区间。

使用 vgg_style_hed 的时候，L2 regularizer 也是可用可不用

使用 vgg_style_hed 的时候推荐使用下面这种配置方式，也就是要使用 batch norm
const.use_batch_norm = True 
const.use_kernel_regularizer = False
'''




'''
# 可以把目录和文件路径定义在 const 里，并且在这里创建必要的目录，
# 也可以像这分代码这样，在每个 python 里，通过命令行参数指定必要的目录或文件，
# 两种手段各有优劣，按需使用就行。

类似下面这种代码

const.data_dir = './data/hed_images'

const.training_data_dir = os.path.join(const.data_dir, 'training')
const.validation_data_dir = os.path.join(const.data_dir, 'validation')
const.test_data_dir = os.path.join(const.data_dir, 'test')

const.training_data_file_path = os.path.join(const.data_dir, 'training_data.csv')
const.validation_data_file_path = os.path.join(const.data_dir, 'validation_data.csv')
const.test_data_file_path = os.path.join(const.data_dir, 'test_data.csv')

const.model_dir = './model'
const.log_dir = './log'

const.checkpoint_dir = os.path.join(const.model_dir, 'checkpoint')
const.checkpoint_file_path = os.path.join(const.checkpoint_dir, 'checkpoint.ckpt')

all_directorys = [const.data_dir,
                  const.training_data_dir,
                  const.validation_data_dir,
                  const.test_data_dir,
                  const.model_dir,
                  const.log_dir
                 ]

for dir in all_directorys:
    if not os.path.exists(dir):
        os.makedirs(dir)
'''