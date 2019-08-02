# HED-tutorial-for-document-scanning
Code for blog [手机端运行卷积神经网络实现文档检测功能(二) -- 从 VGG 到 MobileNetV2 知识梳理](http://fengjian0106.github.io/2018/06/02/Document-Scanning-With-TensorFlow-And-OpenCV-Part-Two/)

## get code
```
git clone https://github.com/fengjian0106/hed-tutorial-for-document-scanning
```

## how to run
#### _1_ Prepare image resources, synthesize training samples

_1.1_ background image is downloaded to the ./sample\_images/background\_images directory.  

_1.2_ The foreground image is downloaded to the ./sample\_images/rect\_images directory.

#### 2 Synthesize training samples using the iPhone simulator
_2.1_ Open the ./generate\_training\_dataset/generate\_training\_dataset.xcodeproj project.

- Check the loadImagePaths function of ViewController.m: make sure that self.backgroundImagesPath and self.rectImagesPath point to the directories mentioned in 1.1 and 1.2 respectively, then run The program (_dataset.xcodeproj project) by pressing Run button on the xcode. Choose an iphone simulator to run this code (Please dont run this code on apple device) , and based on the printed log information, find the directory corresponding to self.imageSaveFolder on the Mac, the generated sample image will be saved in this directory.

- Basically you will observe two types of images : image1_color.jpg, image1_annotation.png. image2_color.jpg, image2_annotation.png ... etc. (I have attached few images in ```>> dataset->generate_sample_by_ios_image_size_256_256_thickness_0.2  (folder)``` for my reference)

_2.2_ Move all the images generated in 2.1 to the ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2 directory.

_2.3_ The white rectangle border drawn on the UIView is smoothed. The white *Point* corresponds to a pixel value other than 255. Therefore, you need to binarize these white *Point*s and run the following program. :

```
python preprocess_generate_training_dataset.py \
			--dataset_root_dir dataset \
			--dataset_folder_name generate_sample_by_ios_image_size_256_256_thickness_0.2
```                                        

After this program is executed, you will get the ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2.csv file.

_2.4_ Use the *gshuf* tool to randomly mess up the contents of the ./dataset/generate\_sample\_by\_ios\_image\_size\_256\_256\_thickness\_0.2.csv file and execute the following command:

```
gshuf ./dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv > ./dataset/temp.txt
gshuf ./dataset/temp.txt > ./dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv
```

After performing this step, a batch of synthetic training sample images is obtained.

The process of preparing the training samples should be customized according to the specific needs. Here is just a reference method. 

#### _3_ Training Network
Run the following program:

```
python train_hed.py --dataset_root_dir dataset \
                    --csv_path dataset/generate_sample_by_ios_image_size_256_256_thickness_0.2.csv \
                    --display_step 5
```
After the training is done you get additional two types of images in the dataset->generate_sample_by_ios_image_size_256_256_thickness_0.2 folder these new images have names image1_annotation_gray.png and image1_annotation_thresh_gray.png. image2_annotation_gray.png and image2_annotation_thresh_gray.png ... etc

#### _4_ Testing HED networks in a python environment
Run the following program to process a picture:

```
python evaluate_hed.py --checkpoint_dir checkpoint \
                       --image test_image/test27.jpg \
                       --output_dir test_image
```

#### _5_ Run a complete process in the iPhone real-world environment, including running the HED network and executing an OpenCV-based point-finding algorithm
_5.1_ Export the model file in pb format and run the following program:

```
python freeze_model.py --checkpoint_dir checkpoint
```

After running successfully, you can see a model file named *hed_graph.pb* in the ./checkpoint directory, which will be loaded in the iOS program.

_5.2_ Running the iOS demo program

*./ios\_demo/DemoWithStaticLib/DemoWithStaticLib.xcodeproj* is a demo program that contains compiled static libraries of various dependencies and can be run directly. There is a complete process in the demo. The first step is to call the HED network to get the edge detection map. The second step is to execute the algorithm for finding the quadrilateral vertices.

_5.3_ Compile FMHEDNet static library

*./ios\_demo/FMHEDNet/FMHEDNet.xcodeproj* is a static library project that encapsulates the call to the HED network to avoid introducing TensorFlow source files into the project files of the business layer app. If you want to compile this FMHEDNet static library, you need to compile TensorFlow Mobile first. For how to compile TensorFlow Mobile, please see _5.4_ below. For the process of compiling FMHEDNet, please see [here] (https://github.com/fengjian0106/hed-tutorial-for-document-scanning/blob/master/ios_demo/FMHEDNet/FMHEDNet/FMHEDNet.mm).

_5.4_ Compile TensorFlow Mobile
TensorFlow's [official documentation] (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile) has instructions for compiling. I am using a manually cropped version and have modified the namespace in the Protobuf source. For details, see [here] (https://github.com/fengjian0106/hed-tutorial-for-document-scanning/blob/master /how_to_build_tensorflow_and_change_namespace_of_protobuf.txt).

