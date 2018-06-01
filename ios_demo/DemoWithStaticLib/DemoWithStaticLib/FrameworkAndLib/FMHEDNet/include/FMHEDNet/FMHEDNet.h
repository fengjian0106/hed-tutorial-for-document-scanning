//
//  FMHEDNet.h
//  FMHEDNet
//
//  Created by fengjian on 2018/4/9.
//  Copyright © 2018年 fengjian. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <opencv2/opencv.hpp>

FOUNDATION_EXPORT NSString * const FMHEDNetErrorDomain;
FOUNDATION_EXPORT NSString * const FMHEDNetProcessImageErrorKey;

typedef NS_ENUM(NSUInteger, FMHEDNetErrorCode) {
    FMHEDNetModelLoadError,
    FMHEDNetProcessImageError,
};


@interface FMHEDNet : NSObject
@property (nonatomic, assign, readonly) BOOL loadModelSuccess;
+ (int)inputImageHeight;
+ (int)inputImageWidth;
+ (int)inputImageChannels;

- (instancetype)initWithModelPath:(NSString *)modelPath;

/**
 inputImage's type must be CV_32FC3, and follow channel order of RGB, pixel value range from 0.0 to 1.0
 outputImage's type will be CV_32FC1, and pixel value range from 0.0 to 1.0
 */
- (BOOL)processImage:(const cv::Mat&)inputImage
         outputImage:(cv::Mat&)outputImage
               error:(NSError * __autoreleasing *)error;
    
@end
