# hed-tutorial-for-document-scanning
Code for blog [手机端运行卷积神经网络实现文档检测功能(二) -- 从 VGG 到 MobileNetV2 知识梳理](http://fengjian0106.github.io/2018/06/02/Document-Scanning-With-TensorFlow-And-OpenCV-Part-Two/)

## get code
```
git clone https://github.com/fengjian0106/hed-tutorial-for-document-scanning
```

## how to run
#### _1_ 准备图片资源，合成训练样本
_1.1_ 背景图片下载到 ./sample\_images/background\_images 目录。  

_1.2_ 前景图片下载到 ./sample\_images/rect\_images 目录。  

#### _2_ 使用 iPhone 模拟器合成训练样本
_2.1_ 打开 ./generate\_training\_dataset/generate\_training\_dataset.xcodeproj 工程，先检查 ViewController.m 的 loadImagePaths 函数，确保 self.backgroundImagesPath 和 self.rectImagesPath 分别指向了 1.1 和 1.2 对应的目录，然后运行程序，并且根据打印的日志信息，在 Mac 上找到 self.imageSaveFolder 对应的目录，生成的样本图片就将保存在这个目录里。  

_2.2_ 将 2.1 里面生成的图片，全部移动到 ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2 目录里。  

_2.3_ 在 UIView 上绘制的白色矩形边框，是有平滑处理的，白色的 *Point* 对应的 像素数值并不是 255，所以还需要对这些白色的 *Point* 进行二值化处理，运行如下程序:

```
python preprocess_generate_training_dataset.py \
			--dataset_root_dir dataset \
			--dataset_folder_name generate_sample_by_ios_image_size_256_256_thickness_0.2
```                                        

这个程序执行完毕后，会得到 ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2.csv 文件。

_2.4_ 利用 *gshuf* 工具，随机打乱 ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2.csv 文件的内容，执行如下命令:

```
gshuf ./dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv > ./dataset/temp.txt
gshuf ./dataset/temp.txt > ./dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv
```

执行到这一步，就得到了一批合成的训练样本图片。  

准备训练样本的过程，应该根据具体的需求定制化开发，这里给的只是一种参考方式。比如还可以人工标注一批图片，也按照同样的格式组织到 csv 文件里。

#### _3_ 训练网络
运行如下程序:

```
python train_hed.py --dataset_root_dir dataset \
                    --csv_path dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv \
                    --display_step 5
```


#### _4_ 在 python 环境中测试 HED 网络
运行如下程序，处理一张图片:

```
python evaluate_hed.py --checkpoint_dir checkpoint \
                       --image test_image/test27.jpg \
                       --output_dir test_image
```

#### _5_ 在 iPhone 真机环境，运行完整的流程，包括运行 HED 网络和执行基于 OpenCV 实现的找点算法
_5.1_ 导出 pb 格式的模型文件，运行如下程序:

```
python freeze_model.py --checkpoint_dir checkpoint
```

成功运行后，可以在 ./checkpoint 目录里看到一个名为 *hed_graph.pb* 的模型文件，iOS 程序中会加载这个模型文件。

_5.2_ 运行 iOS demo 程序  

*./ios\_demo/DemoWithStaticLib/DemoWithStaticLib.xcodeproj* 是一个 demo 程序，工程里面已经包含了编译好的各种依赖的静态库，可以直接运行。demo 里面有完整的流程，第一步是调用 HED 网络得到边缘检测图，第二步是执行找四边形顶点的算法。

_5.3_ 编译 FMHEDNet 静态库

*./ios\_demo/FMHEDNet/FMHEDNet.xcodeproj* 是一个静态库工程项目，里面封装了对 HED 网络的调用过程，这样可以避免在业务层 app 的工程文件中引入 TensorFlow 的源码文件。如果想编译这个 FMHEDNet 静态库，需要先编译 TensorFlow Mobile，关于如何编译 TensorFlow Mobile，请看后面的 _5.4_ 。编译 FMHEDNet 的流程，请看[这里](https://github.com/fengjian0106/hed-tutorial-for-document-scanning/blob/master/ios_demo/FMHEDNet/FMHEDNet/FMHEDNet.mm)。

_5.4_ 编译 TensorFlow Mobile
TensorFlow 的[官方文档](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)有介绍编译的步骤。我使用的是手动裁剪过的版本，并且修改过 Protobuf 源码中的 namespace，具体步骤请看[这里](https://github.com/fengjian0106/hed-tutorial-for-document-scanning/blob/master/how_to_build_tensorflow_and_change_namespace_of_protobuf.txt)。

