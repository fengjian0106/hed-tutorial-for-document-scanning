//
//  ViewController.h
//  generate_training_dataset
//
//  Created by fengjian on 2017/3/29.
//  Copyright © 2017年 fengjian. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController


@end


@interface UIImage (Utilities)
+ (UIImage *)imageWithView:(UIView *)view;
+ (UIImage *)imageWithView_version2:(UIView *)view;
- (UIImage *)scaleToSize:(CGSize)size;
- (UIImage *)crop:(CGRect)rect;

//https://github.com/TimOliver/TOCropViewController/blob/master/TOCropViewController/Models/UIImage%2BCropRotate.m
- (BOOL)hasAlpha;
- (UIImage *)croppedImageWithFrame:(CGRect)frame angle:(NSInteger)angle circularClip:(BOOL)circular;//旋转+裁剪+循环补充被截掉的内容
////////////////////////////////////
@end



@interface NSString (Utilities)
+ (NSString *)randomStringWithLength:(int)length;
@end
