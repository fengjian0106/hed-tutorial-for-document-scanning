//
//  ViewController.m
//  DemoWithStaticLib
//
//  Created by fengjian on 2018/4/9.
//  Copyright © 2018年 fengjian. All rights reserved.
//

#import "ViewController.h"
#import <opencv2/highgui/cap_ios.h>
#import <FMHEDNet/FMHEDNet.h>
#import <FMHEDNet/fm_ocr_scanner.hpp>
#import "OpenCVUtil.h"

//如果不使用视频流，只使用单独的一张图片进行测试，则打开下面这个宏
//#define DEBUG_SINGLE_IMAGE

#define VIDEO_SIZE AVCaptureSessionPreset640x480
#define HW_RATIO (64.0/48.0)

//#define LOG_CV_MAT_TYPE(mat) NSLog(@"___log_OpenCV_info___, "#mat".type() is: %d", mat.type());
#define LOG_CV_MAT_TYPE(mat)


@interface ViewController () <CvVideoCameraDelegate>

@property (weak, nonatomic) IBOutlet UIView *rawVideoView;
@property (weak, nonatomic) IBOutlet UIImageView *imageView1;
@property (weak, nonatomic) IBOutlet UIImageView *imageView2;
@property (weak, nonatomic) IBOutlet UIImageView *imageView3;
@property (weak, nonatomic) IBOutlet UIImageView *imageView4;
@property (weak, nonatomic) IBOutlet UILabel *infoLabel;

@property (nonatomic, assign) BOOL isDebugMode;
@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (nonatomic, strong) FMHEDNet *hedNet;
@property (nonatomic, assign) NSTimeInterval timestampForCallProcessImage;

#ifdef DEBUG_SINGLE_IMAGE
@property (nonatomic, assign) cv::Mat inputImageMat;
#endif
@end



@implementation ViewController
- (CvVideoCamera *)videoCamera {
    if (!_videoCamera) {
        _videoCamera = [[CvVideoCamera alloc] initWithParentView:self.rawVideoView];
        _videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        _videoCamera.defaultAVCaptureSessionPreset = VIDEO_SIZE;
        _videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        _videoCamera.rotateVideo = YES;
        _videoCamera.defaultFPS = 30;
        _videoCamera.grayscaleMode = NO;
        _videoCamera.delegate = self;
    }
    
    return _videoCamera;
}

- (FMHEDNet *)hedNet {
    if (!_hedNet) {
        NSString* const modelFileName = @"hed_graph";
        NSString* const modelFileType = @"pb";
        NSString* modelPath = [[NSBundle mainBundle] pathForResource:modelFileName ofType:modelFileType];
        NSLog(@"---- modelPath is: %@", modelPath);
        _hedNet = [[FMHEDNet alloc] initWithModelPath:modelPath];
        NSLog(@"---- _hedNet is: %@", _hedNet);
    }
    
    return _hedNet;
}

- (void)setIsDebugMode:(BOOL)isDebugMode {
    _isDebugMode = isDebugMode;
    
    if (_isDebugMode) {
        self.rawVideoView.hidden = YES;
        self.imageView1.hidden = NO;
        self.imageView2.hidden = NO;
        self.imageView3.hidden = NO;
        self.imageView4.hidden = NO;
    } else {
        self.rawVideoView.hidden = NO;
        self.imageView1.hidden = YES;
        self.imageView2.hidden = YES;
        self.imageView3.hidden = YES;
        self.imageView4.hidden = YES;
    }
}
    

    
- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    self.isDebugMode = NO;
    
#ifdef DEBUG_SINGLE_IMAGE
    self.isDebugMode = YES;
    
    UIImage *inputImage = [UIImage imageNamed:@"test_image.jpg"];
    self.inputImageMat = [OpenCVUtil cvMatFromUIImage:inputImage];
#endif
    
    NSLog(@"--debug, opencv version is: %s", CV_VERSION);
}
    
- (void)viewWillLayoutSubviews {
    CGFloat containerViewWidth = self.view.frame.size.width;
    CGFloat imageViewWidth = containerViewWidth / 2;
    CGFloat topPadding = 50.0;
    
    self.rawVideoView.frame = CGRectMake(0.0, 0.0 + topPadding, containerViewWidth, containerViewWidth * HW_RATIO);
    
    self.imageView1.frame = CGRectMake(0.0, 0.0 + topPadding,
                                       imageViewWidth, imageViewWidth * HW_RATIO);
    self.imageView2.frame = CGRectMake(containerViewWidth / 2, 0.0 + topPadding,
                                       imageViewWidth, imageViewWidth * HW_RATIO);
    
    self.imageView3.frame = CGRectMake(0.0, imageViewWidth * HW_RATIO + topPadding,
                                       imageViewWidth, imageViewWidth * HW_RATIO);
    self.imageView4.frame = CGRectMake(containerViewWidth / 2, imageViewWidth * HW_RATIO + topPadding,
                                       imageViewWidth, imageViewWidth * HW_RATIO);
}

- (void)viewWillAppear:(BOOL)animated {
    [self startCapture];
}
    
- (void)viewWillDisappear:(BOOL)animated {
    [self stopCapture];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}
    
- (IBAction)changeMode:(id)sender {
    self.isDebugMode = !self.isDebugMode;
}


- (void)startCapture {
    self.timestampForCallProcessImage = [[NSDate date] timeIntervalSince1970];
    [self.videoCamera start];
}
    
- (void)stopCapture {
    [self.videoCamera stop];
}

- (void)debugShowFloatCVMatPixel:(cv::Mat&)mat {
    int height = mat.rows;
    int width = mat.cols;
    int depth = mat.channels();
    
    const float *source_data = (float*) mat.data;
    
    for (int y = 0; y < height; ++y) {
        const float* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const float* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const float* source_value = source_pixel + c;
                
                NSLog(@"-- *source_value is: %f", *source_value);
            }
        }
    }
}

- (UIImage *)imageWithImage:(UIImage *)image scaledToSize:(CGSize)newSize {
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

- (void)processImage:(cv::Mat&)bgraImage {
    //NSLog(@"%s", __PRETTY_FUNCTION__);
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
    
    /**
     2018-04-17 16:56:22.993532+0800 DemoWithStaticLib[945:184826] ___log_OpenCV_info___, rawBgraImage.type() is: 24
     2018-04-17 16:56:22.995671+0800 DemoWithStaticLib[945:184826] ___log_OpenCV_info___, hedSizeOriginalImage.type() is: 24
     2018-04-17 16:56:22.995895+0800 DemoWithStaticLib[945:184826] ___log_OpenCV_info___, rgbImage.type() is: 16
     2018-04-17 16:56:22.996490+0800 DemoWithStaticLib[945:184826] ___log_OpenCV_info___, floatRgbImage.type() is: 21
     2018-04-17 16:56:23.082157+0800 DemoWithStaticLib[945:184826] ___log_OpenCV_info___, hedOutputImage.type() is: 5
    */
#ifdef DEBUG_SINGLE_IMAGE
    cv::Mat rawBgraImage = self.inputImageMat;
#else
    cv::Mat& rawBgraImage = bgraImage;
#endif
    
    //
    LOG_CV_MAT_TYPE(rawBgraImage);
    assert(rawBgraImage.type() == CV_8UC4);
    
    
    // resize rawBgraImage HED Net size
    int height = [FMHEDNet inputImageHeight];
    int width = [FMHEDNet inputImageWidth];
    cv::Size size(width, height);
    cv::Mat hedSizeOriginalImage;
    cv::resize(rawBgraImage, hedSizeOriginalImage, size, 0, 0, cv::INTER_LINEAR);
    LOG_CV_MAT_TYPE(hedSizeOriginalImage);
    assert(hedSizeOriginalImage.type() == CV_8UC4);
    
    
    // convert from BGRA to RGB
    cv::Mat rgbImage;
    cv::cvtColor(hedSizeOriginalImage, rgbImage, cv::COLOR_BGRA2RGB);
    LOG_CV_MAT_TYPE(rgbImage);
    assert(rgbImage.type() == CV_8UC3);
    
    
    // convert pixel type from int to float, and value range from (0, 255) to (0.0, 1.0)
    cv::Mat floatRgbImage;
    /**
     void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;
     */
    rgbImage.convertTo(floatRgbImage, CV_32FC3, 1.0 / 255);
    LOG_CV_MAT_TYPE(floatRgbImage);
    /**
     floatRgbImage 是归一化处理后的矩阵，
     如果使用 VGG style HED，并且没有使用 batch norm 技术，那就不需要做归一化处理，
     而是参照 VGG 的使用惯例，减去像素平均值，类似下面的代码
     //http://answers.opencv.org/question/59529/how-do-i-separate-the-channels-of-an-rgb-image-and-save-each-one-using-the-249-version-of-opencv/
     //http://opencvexamples.blogspot.com/2013/10/split-and-merge-functions.html
     const float R_MEAN = 123.68;
     const float G_MEAN = 116.78;
     const float B_MEAN = 103.94;
     
     cv::Mat rgbChannels[3];
     cv::split(floatRgbImage, rgbChannels);
     
     rgbChannels[0] = rgbChannels[0] - R_MEAN;
     rgbChannels[1] = rgbChannels[1] - G_MEAN;
     rgbChannels[2] = rgbChannels[2] - B_MEAN;
     
     std::vector<cv::Mat> channels;
     channels.push_back(rgbChannels[0]);
     channels.push_back(rgbChannels[1]);
     channels.push_back(rgbChannels[2]);
     
     cv::Mat vggStyleImage;
     cv::merge(channels, vggStyleImage);
     */

    
    // run hed net
    cv::Mat hedOutputImage;
    NSError *error;
    NSTimeInterval startTime = [[NSDate date] timeIntervalSince1970];
    if ([self.hedNet processImage:floatRgbImage outputImage:hedOutputImage error:&error]) {
        LOG_CV_MAT_TYPE(hedOutputImage);
        NSTimeInterval hedTime = [[NSDate date] timeIntervalSince1970] - startTime;
        
        
        startTime = [[NSDate date] timeIntervalSince1970];
        auto tuple = ProcessEdgeImage(hedOutputImage, rgbImage, self.isDebugMode);
        NSTimeInterval opencvTime = [[NSDate date] timeIntervalSince1970] - startTime;
        
        // FPS
        NSTimeInterval lasTimestamp = self.timestampForCallProcessImage;
        self.timestampForCallProcessImage = [[NSDate date] timeIntervalSince1970];
        NSUInteger FPS = (NSUInteger)(1.0 / (self.timestampForCallProcessImage - lasTimestamp));
        
        
        NSString *debugInfo = [NSString stringWithFormat:@"hed time: %.7f second\nopencv time: %.7f second\ntotal FPS: %lu", hedTime, opencvTime, (unsigned long)FPS];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.infoLabel.text = debugInfo;
        });
        
        auto find_rect = std::get<0>(tuple);
        auto cv_points = std::get<1>(tuple);
        auto debug_mats = std::get<2>(tuple);
        

        if (self.isDebugMode) {
            UIImage *image1 = [OpenCVUtil UIImageFromCVMat:debug_mats[0].clone()];
            UIImage *image2 = [OpenCVUtil UIImageFromCVMat:debug_mats[1].clone()];
            UIImage *image3 = [OpenCVUtil UIImageFromCVMat:debug_mats[2].clone()];
            UIImage *image4 = [OpenCVUtil UIImageFromCVMat:debug_mats[3].clone()];
            
            dispatch_async(dispatch_get_main_queue(), ^{
                self.imageView1.image = [self imageWithImage:image1 scaledToSize:self.imageView1.frame.size];
                self.imageView2.image = [self imageWithImage:image2 scaledToSize:self.imageView2.frame.size];
                self.imageView3.image = [self imageWithImage:image3 scaledToSize:self.imageView3.frame.size];
                self.imageView4.image = [self imageWithImage:image4 scaledToSize:self.imageView4.frame.size];
            });
        } else {
            if (find_rect == true) {
                std::vector<cv::Point> scaled_points;
                int original_height, original_width;
                original_height = rawBgraImage.rows;
                original_width = rawBgraImage.cols;
                
                for(int i = 0; i < cv_points.size(); i++) {
                    cv::Point cv_point = cv_points[i];
                    
                    cv::Point scaled_point = cv::Point(cv_point.x * original_width / [FMHEDNet inputImageWidth], cv_point.y * original_height / [FMHEDNet inputImageHeight]);
                    scaled_points.push_back(scaled_point);
                    
                    /** convert from cv::Point to CGPoint
                     CGPoint point = CGPointMake(scaled_point.x, scaled_point.y);
                    */
                }
                
                
                cv::line(rawBgraImage, scaled_points[0], scaled_points[1], CV_RGB(255, 0, 0), 2);
                cv::line(rawBgraImage, scaled_points[1], scaled_points[2], CV_RGB(255, 0, 0), 2);
                cv::line(rawBgraImage, scaled_points[2], scaled_points[3], CV_RGB(255, 0, 0), 2);
                cv::line(rawBgraImage, scaled_points[3], scaled_points[0], CV_RGB(255, 0, 0), 2);
            }
        }
    } else {
        NSLog(@"hedNet processImage error: %@", error);
    }
}

@end
