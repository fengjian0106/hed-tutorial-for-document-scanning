//
//  FMHEDNet.m
//  FMHEDNet
//
//  Created by fengjian on 2018/4/9.
//  Copyright © 2018年 fengjian. All rights reserved.
//


/**
 <1>
 how to link TensorFlow static lib -- https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios
 
 You'll need to update various settings in your app to link against TensorFlow. You can view them in the example projects, but here's a full rundown:
 
 The compile_ios_tensorflow.sh script builds a universal static library in tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a. You'll need to add this to your linking build stage, and in Search Paths add tensorflow/contrib/makefile/gen/lib to the Library Search Paths setting.
 
 You'll also need to add libprotobuf.a and libprotobuf-lite.a from tensorflow/contrib/makefile/gen/protobuf_ios/lib and nsync.a from tensorflow/contrib/makefile/downloads/nsync/builds/lipo.ios.c++11 to your Build Stages and Library Search Paths.
 
 The Header Search paths needs to contain:
 
 the root folder of tensorflow,
 tensorflow/contrib/makefile/downloads/nsync/public
 tensorflow/contrib/makefile/downloads/protobuf/src
 tensorflow/contrib/makefile/downloads,
 tensorflow/contrib/makefile/downloads/eigen, and
 tensorflow/contrib/makefile/gen/proto.
 In the Linking section, you need to add -force_load followed by the path to the TensorFlow static library in the Other Linker Flags section. This ensures that the global C++ objects that are used to register important classes inside the library are not stripped out. To the linker, they can appear unused because no other code references the variables, but in fact their constructors have the important side effect of registering the class.
 
 You'll need to include the Accelerate framework in the "Link Binary with Libraries" build phase of your project.
 
 C++11 support (or later) should be enabled by setting C++ Language Dialect to GNU++11 (or GNU++14), and C++ Standard Library to libc++.
 
 The library doesn't currently support bitcode, so you'll need to disable that in your project settings.
 
 Remove any use of the -all_load flag in your project. The protocol buffers libraries (full and lite versions) contain duplicate symbols, and the -all_load flag will cause these duplicates to become link errors. If you were using -all_load to avoid issues with Objective-C categories in static libraries, you may be able to replace it with the -ObjC flag.
 
 <2>
 this project is just a static lib, so no need to link TensorFlow static lib, just set TensorFlow header path
 
 e.g.
 
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace/tensorflow/contrib/makefile/downloads/nsync/public/
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace/tensorflow/contrib/makefile/downloads/protobuf/src
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace/tensorflow/contrib/makefile/downloads
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace/tensorflow/contrib/makefile/downloads/eigen
 /Users/fengjian/my-work/machine-learning/TensorFlow_source_code/tensorflow-1.7.0-change-protobuf-namespace/tensorflow/contrib/makefile/gen/proto

 !! change to your TensorFlow source code root path
 
 
 <3>
 how to make universal static library -- ref https://www.raywenderlich.com/41377/creating-a-static-library-in-ios-tutorial
 
 and below is my version
 
 '''
 # define output folder environment variable
 UNIVERSAL_OUTPUTFOLDER=${BUILD_DIR}/${CONFIGURATION}-universal
 
 # build device and simulator versions
 ## xcodebuild -target ${PROJECT_NAME} ONLY_ACTIVE_ARCH=NO -configuration ${CONFIGURATION} -sdk iphoneos BUILD_DIR="${BUILD_DIR}" BUILD_ROOT="${BUILD_ROOT}"
 xcodebuild -target ${PROJECT_NAME} ONLY_ACTIVE_ARCH=NO -configuration ${CONFIGURATION} -sdk iphoneos -arch armv7 -arch armv7s -arch arm64 BUILD_DIR="${BUILD_DIR}" BUILD_ROOT="${BUILD_ROOT}"
 xcodebuild -target ${PROJECT_NAME} ONLY_ACTIVE_ARCH=NO -configuration ${CONFIGURATION} -sdk iphonesimulator BUILD_DIR="${BUILD_DIR}" BUILD_ROOT="${BUILD_ROOT}"
 
 # make sure the output directory exists
 mkdir -p "${UNIVERSAL_OUTPUTFOLDER}"
 
 # create universal binary file using lipo
 lipo -create -output "${UNIVERSAL_OUTPUTFOLDER}/lib${PROJECT_NAME}.a" "${BUILD_DIR}/${CONFIGURATION}-iphoneos/lib${PROJECT_NAME}.a" "${BUILD_DIR}/${CONFIGURATION}-iphonesimulator/lib${PROJECT_NAME}.a"
 
 echo "Universal library can be found here:"
 echo ${UNIVERSAL_OUTPUTFOLDER}/lib${PROJECT_NAME}.a
 
 # copy the header files to the final output folder
 cp -R "${BUILD_DIR}/${CONFIGURATION}-iphoneos/include" "${UNIVERSAL_OUTPUTFOLDER}/"
 
 # remove the build folder
 rm -rf ${SRCROOT}/build
 '''
 
 !! attention !!
 在 XCode9.3 中，
 set 'iOS Deployment Target' in project to iOS 8.0
 才能同时编译出 arm64、armv7 和 armv7s 三个版本
 否则只会有 arm64 和 armv7s 两个版本
 !! attention !!
 */

#import "FMHEDNet.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"

/**
 调试的时候想看不同的 layer 消耗的 cpu、memory 等信息，打开 TRACE_TF 这个宏就可以了，输出的内容类似下面这样
 
 2018-05-16 16:35:20.840543: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Run Order ==============================
 2018-05-16 16:35:20.840558: I tensorflow/core/util/stat_summarizer.cc:468]                  [node type]      [start]      [first]     [avg ms]         [%]      [cdf%]      [mem KB]    [times called]    [Name]
 2018-05-16 16:35:20.840565: I tensorflow/core/util/stat_summarizer.cc:468]                         NoOp        0.000        0.838        0.838      0.918%      0.918%         0.000            1    _SOURCE
 2018-05-16 16:35:20.840613: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.859        0.018        0.018      0.020%      0.937%         0.000            1    hed/mobilenet_v2/block5_1/projection_1x1_conv2d/batch_normalization/moving_variance/read/_242__cf__242
 2018-05-16 16:35:20.840620: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.880        0.004        0.004      0.004%      0.942%         0.000            1    hed/mobilenet_v2/block5_1/projection_1x1_conv2d/conv2d/kernel/read/_243__cf__243
 2018-05-16 16:35:20.840626: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.886        0.003        0.003      0.003%      0.945%         0.000            1    hed/mobilenet_v2/block5_2/depthwise_conv2d/SeparableConv2d/depthwise_weights/read/_244__cf__244
 2018-05-16 16:35:20.840632: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.891        0.003        0.003      0.003%      0.948%         0.000            1    hed/mobilenet_v2/block5_2/depthwise_conv2d/batch_normalization/beta/read/_245__cf__245
 2018-05-16 16:35:20.840675: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.895        0.003        0.003      0.003%      0.951%         0.000            1    hed/mobilenet_v2/block5_2/depthwise_conv2d/batch_normalization/gamma/read/_246__cf__246
 2018-05-16 16:35:20.840681: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.899        0.003        0.003      0.003%      0.955%         0.000            1    hed/mobilenet_v2/block5_2/depthwise_conv2d/batch_normalization/moving_mean/read/_247__cf__247
 2018-05-16 16:35:20.840687: I tensorflow/core/util/stat_summarizer.cc:468]                        Const        0.961        0.004        0.004      0.004%      0.959%         0.000            1    hed/mobilenet_v2/block5_2/depthwise_conv2d/batch_normalization/moving_variance/read/_248__cf__248
 ...............................................................................Run Order 这部分的内容特别的多，下面的那些统计信息更容易看到全局的性能消耗....................................................................................................
 
 
 2018-05-16 16:35:32.501131: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Top by Computation Time ==============================
 2018-05-16 16:35:32.501144: I tensorflow/core/util/stat_summarizer.cc:468]                  [node type]      [start]      [first]     [avg ms]         [%]      [cdf%]      [mem KB]    [times called]    [Name]
 2018-05-16 16:35:32.501158: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       39.578        0.208       18.020      9.156%      9.156%       262.144            1    hed/dsn1/conv2d/Conv2D
 2018-05-16 16:35:32.501172: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       19.133       10.755       15.461      7.856%     17.012%      1572.864            1    hed/mobilenet_v2/block0_1/conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501212: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D        2.789        7.446       12.654      6.430%     23.442%       786.432            1    hed/mobilenet_v2/block0_0/conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501227: I tensorflow/core/util/stat_summarizer.cc:468]               FusedBatchNorm       57.645        0.693        9.022      4.584%     28.026%         0.016            1    hed/dsn1/batch_normalization/cond/FusedBatchNorm_1
 2018-05-16 16:35:32.501241: I tensorflow/core/util/stat_summarizer.cc:468]               FusedBatchNorm       68.847        2.496        6.707      3.408%     31.434%         0.576            1    hed/mobilenet_v2/block2_0/expansion_1x1_conv2d/batch_normalization/cond/FusedBatchNorm_1
 2018-05-16 16:35:32.501255: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       40.282        3.621        5.456      2.772%     34.206%       786.432            1    hed/mobilenet_v2/block0_2/conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501269: I tensorflow/core/util/stat_summarizer.cc:468]               FusedBatchNorm       34.677        1.602        4.342      2.206%     36.412%         0.096            1    hed/mobilenet_v2/block0_1/conv2d/batch_normalization/cond/FusedBatchNorm_1
 2018-05-16 16:35:32.501283: I tensorflow/core/util/stat_summarizer.cc:468]        DepthwiseConv2dNative       55.342        1.525        4.318      2.194%     38.607%       789.888            1    hed/mobilenet_v2/block1_0/depthwise_conv2d/SeparableConv2d/depthwise
 2018-05-16 16:35:32.501297: I tensorflow/core/util/stat_summarizer.cc:468]        DepthwiseConv2dNative       86.945        1.601        4.278      2.174%     40.781%      1221.120            1    hed/mobilenet_v2/block2_1/depthwise_conv2d/SeparableConv2d/depthwise
 2018-05-16 16:35:32.501325: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput      101.296        1.307        4.177      2.122%     42.903%      1310.720            1    hed/dsn3/conv2d_transpose/conv2d_transpose
 2018-05-16 16:35:32.501338: I tensorflow/core/util/stat_summarizer.cc:468]
 2018-05-16 16:35:32.501350: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Top by Memory Use ==============================
 2018-05-16 16:35:32.501363: I tensorflow/core/util/stat_summarizer.cc:468]                  [node type]      [start]      [first]     [avg ms]         [%]      [cdf%]      [mem KB]    [times called]    [Name]
 2018-05-16 16:35:32.501377: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       67.046        1.932        1.760      0.894%      0.894%      2359.296            1    hed/mobilenet_v2/block2_0/expansion_1x1_conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501391: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       19.133       10.755       15.461      7.856%      8.750%      1572.864            1    hed/mobilenet_v2/block0_1/conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501405: I tensorflow/core/util/stat_summarizer.cc:468]                     ConcatV2      149.508        2.141        2.120      1.077%      9.827%      1310.720            1    hed/dsn_fuse/concat
 2018-05-16 16:35:32.501419: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput      148.325        0.818        0.979      0.497%     10.324%      1310.720            1    hed/dsn5/conv2d_transpose/conv2d_transpose
 2018-05-16 16:35:32.501433: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput      121.815        1.054        3.088      1.569%     11.894%      1310.720            1    hed/dsn4/conv2d_transpose/conv2d_transpose
 2018-05-16 16:35:32.501477: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput      101.296        1.307        4.177      2.122%     14.016%      1310.720            1    hed/dsn3/conv2d_transpose/conv2d_transpose
 2018-05-16 16:35:32.501492: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput       70.636        5.084        3.834      1.948%     15.964%      1310.720            1    hed/dsn2/conv2d_transpose/conv2d_transpose
 2018-05-16 16:35:32.501506: I tensorflow/core/util/stat_summarizer.cc:468]        DepthwiseConv2dNative       86.945        1.601        4.278      2.174%     18.138%      1221.120            1    hed/mobilenet_v2/block2_1/depthwise_conv2d/SeparableConv2d/depthwise
 2018-05-16 16:35:32.501520: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       98.616        0.531        0.787      0.400%     18.538%      1179.648            1    hed/mobilenet_v2/block3_0/expansion_1x1_conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501534: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D       82.634        0.369        0.749      0.380%     18.918%      1179.648            1    hed/mobilenet_v2/block2_1/expansion_1x1_conv2d/conv2d/Conv2D
 2018-05-16 16:35:32.501546: I tensorflow/core/util/stat_summarizer.cc:468]
 2018-05-16 16:35:32.501559: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Summary by node type ==============================
 2018-05-16 16:35:32.501584: I tensorflow/core/util/stat_summarizer.cc:468]                  [Node type]      [count]      [avg ms]        [avg %]        [cdf %]      [mem KB]    [times called]
 2018-05-16 16:35:32.501597: I tensorflow/core/util/stat_summarizer.cc:468]                       Conv2D           37        74.951        38.149%        38.149%     14930.944           37
 2018-05-16 16:35:32.501610: I tensorflow/core/util/stat_summarizer.cc:468]               FusedBatchNorm           49        70.019        35.639%        73.788%        90.720           49
 2018-05-16 16:35:32.501623: I tensorflow/core/util/stat_summarizer.cc:468]        DepthwiseConv2dNative           13        21.752        11.072%        84.860%      6418.752           13
 2018-05-16 16:35:32.501636: I tensorflow/core/util/stat_summarizer.cc:468]          Conv2DBackpropInput            4        12.076         6.147%        91.006%      5242.880            4
 2018-05-16 16:35:32.501649: I tensorflow/core/util/stat_summarizer.cc:468]                        Relu6           26         5.755         2.929%        93.935%         0.000           26
 2018-05-16 16:35:32.501662: I tensorflow/core/util/stat_summarizer.cc:468]                        Const          255         5.056         2.573%        96.509%         0.000          255
 2018-05-16 16:35:32.501675: I tensorflow/core/util/stat_summarizer.cc:468]                     ConcatV2            1         2.119         1.079%        97.587%      1310.720            1
 2018-05-16 16:35:32.501687: I tensorflow/core/util/stat_summarizer.cc:468]                       Switch          246         1.349         0.687%        98.274%         0.000          246
 2018-05-16 16:35:32.501701: I tensorflow/core/util/stat_summarizer.cc:468]                         Relu            3         1.203         0.612%        98.886%         0.000            3
 2018-05-16 16:35:32.501726: I tensorflow/core/util/stat_summarizer.cc:468]                      BiasAdd            5         0.887         0.451%        99.338%         0.000            5
 2018-05-16 16:35:32.501745: I tensorflow/core/util/stat_summarizer.cc:468]                        Merge           49         0.619         0.315%        99.653%         0.196           49
 2018-05-16 16:35:32.501758: I tensorflow/core/util/stat_summarizer.cc:468]                          Add           10         0.570         0.290%        99.943%         0.000           10
 2018-05-16 16:35:32.501771: I tensorflow/core/util/stat_summarizer.cc:468]                         NoOp            1         0.058         0.030%        99.973%         0.000            1
 2018-05-16 16:35:32.501783: I tensorflow/core/util/stat_summarizer.cc:468]                     Identity            1         0.026         0.013%        99.986%         0.000            1
 2018-05-16 16:35:32.501796: I tensorflow/core/util/stat_summarizer.cc:468]                      _Retval            1         0.019         0.010%        99.995%         0.000            1
 2018-05-16 16:35:32.501809: I tensorflow/core/util/stat_summarizer.cc:468]                         _Arg            2         0.009         0.005%       100.000%         0.000            2
 2018-05-16 16:35:32.501832: I tensorflow/core/util/stat_summarizer.cc:468]
 2018-05-16 16:35:32.501845: I tensorflow/core/util/stat_summarizer.cc:468] Timings (microseconds): count=57 first=91334 curr=175081 min=91334 max=280501 avg=196806 std=40922
 2018-05-16 16:35:32.501857: I tensorflow/core/util/stat_summarizer.cc:468] Memory (bytes): count=57 curr=27994212(all same)
 2018-05-16 16:35:32.501900: I tensorflow/core/util/stat_summarizer.cc:468] 703 nodes observed
 
 */
//#define TRACE_TF
#ifdef TRACE_TF
#include "tensorflow/core/util/stat_summarizer.h"
#endif


NSString * const FMHEDNetErrorDomain = @"com.fm.hednet.error";
NSString * const FMHEDNetProcessImageErrorKey = @"com.fm.hednet.error.processImageFail";

/**
 freeze_model.py 里面有打印下面这三个 node 的 name 信息
 
 Input Node is:
    Tensor("hed_input:0", shape=(1, 256, 256, 3), dtype=float32)
    Tensor("is_training:0", dtype=bool)
 Output Node is:
    Tensor("hed/dsn_fuse/conv2d/BiasAdd:0", shape=(1, 256, 256, 1), dtype=float32)
 */
const std::string kInputLayerName = "hed_input:0";
const std::string kIsTrainingName = "is_training:0";
const std::string kOutputLayerName = "hed/dsn_fuse/conv2d/BiasAdd:0";


const int kInputImageHeight = 256;
const int kInputImageWidth = 256;
const int kInputImageChannels = 3;

static tensorflow::Status LoadGraph(std::string graph_file_path, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_path, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_path, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return tensorflow::Status::OK();
}



@interface FMHEDNet () {
    std::unique_ptr<tensorflow::Session> tf_session;
#ifdef TRACE_TF
    std::unique_ptr<tensorflow::StatSummarizer> tf_stat_summarizer;
#endif
}
@property (nonatomic, assign, readwrite) BOOL loadModelSuccess;
@end

@implementation FMHEDNet
    
+ (int)inputImageHeight {
    return kInputImageHeight;
}

+ (int)inputImageWidth {
    return kInputImageWidth;
}

+ (int)inputImageChannels {
    return kInputImageChannels;
}

- (BOOL)initTFSessionWithModelPath:(NSString *)modelPath {
    if (!modelPath) {
        NSLog(@"FMHEDNet initTFSession, modelPath not found: %@", modelPath);
        return NO;
    }

    tensorflow::Status load_graph_status = LoadGraph([modelPath UTF8String], &tf_session);
    if (!load_graph_status.ok()) {
        std::cout << "FMHEDNet initTFSession, LoadGraph error:: " << load_graph_status.ToString();
        return NO;
    }
    
#ifdef TRACE_TF
    //bool show_sizes = false;
    bool show_run_order = false;//true;
    int run_order_limit = 0;
    bool show_time = true;
    int time_limit = 10;
    bool show_memory = true;
    int memory_limit = 10;
    bool show_type = true;
    bool show_summary = true;
    //bool show_flops = false;
    //int warmup_runs = 1;
    
    tensorflow::StatSummarizerOptions stats_options;
    stats_options.show_run_order = show_run_order;
    stats_options.run_order_limit = run_order_limit;
    stats_options.show_time = show_time;
    stats_options.time_limit = time_limit;
    stats_options.show_memory = show_memory;
    stats_options.memory_limit = memory_limit;
    stats_options.show_type = show_type;
    stats_options.show_summary = show_summary;
    
    tf_stat_summarizer.reset(new tensorflow::StatSummarizer(stats_options));
#endif

    return YES;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        _loadModelSuccess = YES;

        if ([self initTFSessionWithModelPath:modelPath] == NO) {
            _loadModelSuccess = NO;
        }
    }
    return self;
}


- (BOOL)processImage:(const cv::Mat&)inputImage
         outputImage:(cv::Mat&)outputImage
               error:(NSError * __autoreleasing *)error {
    /**
     https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
     
     +--------+----+----+----+----+------+------+------+------+
     |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
     +--------+----+----+----+----+------+------+------+------+
     | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
     | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
     | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
     | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
     | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
     | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
     | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
     +--------+----+----+----+----+------+------+------+------+
     */
    assert(inputImage.rows == [FMHEDNet inputImageHeight]);
    assert(inputImage.cols == [FMHEDNet inputImageWidth]);
    assert(inputImage.channels() == [FMHEDNet inputImageChannels]);
    assert(inputImage.type() == CV_32FC3);
    
    
    BOOL isDebug = NO;
    //BOOL isDebug = YES;
    
    if (!self.loadModelSuccess) {
        if (error) {
            *error = [NSError errorWithDomain:FMHEDNetErrorDomain code:FMHEDNetModelLoadError userInfo:nil];
        }
        std::cout << "FMHEDNet processImage, self.loadModelSuccess == NO";
        return NO;
    }
    

    int height = [FMHEDNet inputImageHeight];
    int width = [FMHEDNet inputImageWidth];
    int depth = [FMHEDNet inputImageChannels];
    NSError *processImageError = nil;
    
    // input tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, depth}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    
    tensorflow::Tensor is_training(tensorflow::DT_BOOL, tensorflow::TensorShape());
    is_training.scalar<bool>()() = false;

    // copy data into the corresponding tensor
    const float *source_data = (float*) inputImage.data;
    for (int y = 0; y < height; ++y) {
        const float* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const float* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const float* source_value = source_pixel + c;
                input_tensor_mapped(0, y, x, c) = *source_value;
                
                if (isDebug) {
                    NSLog(@"-- *source_value is: %f", *source_value);
                }
            }
        }
    }

    // session run
    if (tf_session.get() != nullptr) { // std::unique_ptr::get()
        std::vector<tensorflow::Tensor> finalOutput;
        if (isDebug) {
            std::cout << "finalOutput, size=" << finalOutput.size() << std::endl;
        }

#ifdef TRACE_TF
        //https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/benchmark/benchmark_model.cc
        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
        
        tensorflow::RunMetadata run_metadata;
        
        const int64 start_time = tensorflow::Env::Default()->NowMicros();
        tensorflow::Status run_status  = tf_session->Run(run_options,
                                                         {{kInputLayerName, input_tensor}, {kIsTrainingName, is_training}},
                                                         {kOutputLayerName},
                                                         {},
                                                         &finalOutput,
                                                         &run_metadata);
        const int64 end_time = tensorflow::Env::Default()->NowMicros();
        const int64 inference_time_us = end_time - start_time;
        std::cout << "tf_session->Run, inference_time_us: " << inference_time_us << std::endl;
        
        assert(run_metadata.has_step_stats());
        const tensorflow::StepStats& step_stats = run_metadata.step_stats();
        tf_stat_summarizer->ProcessStepStats(step_stats);
        
        tf_stat_summarizer->PrintStepStats();
        //tf_stat_summarizer->PrintOutputs();
#else
        tensorflow::Status run_status  = tf_session->Run({{kInputLayerName, input_tensor}, {kIsTrainingName, is_training}},
                                                         {kOutputLayerName},
                                                         {},
                                                         &finalOutput);
#endif
        
        if (run_status.ok() != true) {
            std::cout << "tf_session->Run error: " << run_status.error_message() << std::endl;
            processImageError = [NSError errorWithDomain:FMHEDNetErrorDomain
                                                    code:FMHEDNetProcessImageError
                                                userInfo:@{FMHEDNetProcessImageErrorKey:[NSString stringWithUTF8String:run_status.error_message().c_str()]}];
        } else {
            tensorflow::Tensor output = std::move(finalOutput.at(0));
            if (isDebug) {
                std::cout << "-- tensorflow::Tensor output dtype() is: " << output.dtype() << std::endl;
                std::cout << "-- tensorflow::Tensor output shape().DebugString() is: " << output.shape().DebugString() << std::endl;
                std::cout << "-- tensorflow::Tensor output dims() is: " << output.dims() << std::endl;
                std::cout << "-- tensorflow::Tensor output dim_size(0) is: " << output.dim_size(0) << std::endl;
                std::cout << "-- tensorflow::Tensor output dim_size(1) is: " << output.dim_size(1) << std::endl;
                std::cout << "-- tensorflow::Tensor output dim_size(2) is: " << output.dim_size(2) << std::endl;
                std::cout << "-- tensorflow::Tensor output dim_size(3) is: " << output.dim_size(3) << std::endl;
                /**
                 -- tensorflow::Tensor output dtype() is: 1
                 -- tensorflow::Tensor output shape().DebugString() is: [1,224,224,1]
                 -- tensorflow::Tensor output dims() is: 4
                 -- tensorflow::Tensor output dim_size(0) is: 1
                 -- tensorflow::Tensor output dim_size(1) is: 256
                 -- tensorflow::Tensor output dim_size(2) is: 256
                 -- tensorflow::Tensor output dim_size(3) is: 1
                 */
            }

            /**
             auto scores = output.flat<float>();

             (lldb) print output
             (tensorflow::Tensor) $2 = {
             shape_ = {
             u_ = {
             buf = {
             [0] = '\x01'
             [1] = '\0'
             [2] = '\xe0'
             [3] = '\0'
             [4] = '\xe0'
             [5] = '\0'
             [6] = '\x01'
             [7] = '\0'
             [8] = '\xd8'
             [9] = '\xae'
             [10] = '\x05'
             [11] = '\x02'
             [12] = '\x01'
             [13] = '\x01'
             [14] = '\x04'
             [15] = '\0'
             }
             unused_aligner = 0x000100e000e00001
             }
             num_elements_ = 50176
             }
             buf_ = 0x0000000170657df0
             }


             (lldb) print scores
             (Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16>) $1 = {
             m_data = 0x0000000106c0c000
             m_dimensions = {
             Eigen::array<long, 1> = {
             __elems_ = ([0] = 50176)
             }
             }
             }
             */


            if (isDebug) {
                std::cout << "debug output.dim_size: " << output.dim_size(1) << ", " << output.dim_size(2) << ", " << output.dim_size(3) << std::endl;
            }
            
            // convert tensorflow::Tensor to cv::Mat Or if you do not want to use cv::Mat, you can convert tensorflow::Tensor to Eigen::Map or Eigen::Matrix
            cv::Mat outputMat = cv::Mat((int)output.dim_size(2), (int)output.dim_size(1), CV_32FC1, output.flat<float>().data());
            if (isDebug) {
                std::cout << "--> outputMat is: " << outputMat << std::endl;
            }

            outputImage = outputMat.clone();
        }
    } else {
        processImageError = [NSError errorWithDomain:FMHEDNetErrorDomain code:FMHEDNetModelLoadError userInfo:nil];
        std::cout << "FMHEDNet processImage, tf_session.get() == nullptr, std::unique_ptr::get() == nullptr";
    }

    if (error && processImageError) {
        *error = processImageError;
    }

    return processImageError == nil ? YES : NO;
}
@end
