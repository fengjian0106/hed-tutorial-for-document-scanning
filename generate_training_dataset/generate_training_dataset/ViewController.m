//
//  ViewController.m
//  generate_training_dataset
//
//  Created by fengjian on 2017/3/29.
//  Copyright © 2017年 fengjian. All rights reserved.
//
#import "ViewController.h"

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
//256 * 256
const NSInteger SAMPLE_IMAGE_WIDTH = 256;
const NSInteger SAMPLE_IMAGE_HEIGHT = 256;

//生成的图像，矩形区域的长或宽，至少占据了四分之一的长度
const NSInteger RECT_IMAGE_MIN_WIDTH = 256 / 4;//56
const NSInteger RECT_IMAGE_MIN_HEIGHT = 256 / 4;//56
//生成的图像，矩形区域的长或宽，最多只能是210pix
const NSInteger RECT_IMAGE_MAX_WIDTH = 245;
const NSInteger RECT_IMAGE_MAX_HEIGHT = 245;
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////




@interface ViewController ()
@property (nonatomic, strong) UIImageView *backgroundImageView;
@property (nonatomic, strong) UIImageView *rectImageView;
@property (nonatomic, strong) UIImageView *edgeImageView;

@property (nonatomic, strong) UILabel *debugLabel;

@property (nonatomic, strong) NSString *backgroundImagesPath;
@property (nonatomic, strong) NSString *rectImagesPath;
@property (nonatomic, strong) NSArray *backgroundImageFileNames;
@property (nonatomic, strong) NSArray *rectImageFileNames;

@property (nonatomic, strong) NSString *imageSaveFolder;
@end

@implementation ViewController



- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    self.backgroundImageView = [[UIImageView alloc] initWithImage:nil];
    [self.view addSubview:self.backgroundImageView];
    self.backgroundImageView.backgroundColor = [UIColor redColor];
    
    self.rectImageView = [[UIImageView alloc] initWithImage:nil];
    [self.backgroundImageView addSubview:self.rectImageView];
    self.rectImageView.backgroundColor = [UIColor greenColor];
    
    self.edgeImageView = [[UIImageView alloc] initWithImage:nil];
    [self.view addSubview:self.edgeImageView];
    self.edgeImageView.backgroundColor = [UIColor blackColor];
    
    self.debugLabel = [[UILabel alloc] init];
    [self.view addSubview:self.debugLabel];
    self.debugLabel.backgroundColor = [UIColor yellowColor];
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    
    [self loadImagePaths];
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self generateDataset];
    });
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)loadImagePaths {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];//生成的 image 保存到这个目录里
    NSLog(@"documentsDirectory is: %@", documentsDirectory);
    self.imageSaveFolder = documentsDirectory;
    
    NSString *resourcePath = [[NSBundle mainBundle] resourcePath];
    NSLog(@"resourcePath is: %@", resourcePath);//沙盒路径
    
    
    self.backgroundImagesPath = [resourcePath stringByAppendingPathComponent:@"background_images"];
    self.rectImagesPath = [resourcePath stringByAppendingPathComponent:@"rect_images"];
    
    NSFileManager *manager = [NSFileManager defaultManager];
    self.backgroundImageFileNames = [manager contentsOfDirectoryAtPath:self.backgroundImagesPath error:nil];
    self.rectImageFileNames = [manager contentsOfDirectoryAtPath:self.rectImagesPath error:nil];
    
    self.backgroundImageFileNames = [self.backgroundImageFileNames subarrayWithRange:NSMakeRange(110, self.backgroundImageFileNames.count - 110)];
    
    /**
    for (NSString *s in self.backgroundImageFileNames){
        NSLog(@"%@", s);
    }
    for (NSString *s in self.rectImageFileNames){
        NSLog(@"%@", s);
    }
    */
}

- (void)viewWillLayoutSubviews {
    CGFloat padding = 20.0;
    CGFloat leftX = (self.view.frame.size.width - SAMPLE_IMAGE_WIDTH) / 2;
    CGFloat leftY1 = (self.view.frame.size.height - SAMPLE_IMAGE_HEIGHT * 2 - padding) / 2;
    CGFloat leftY2 = leftY1 + SAMPLE_IMAGE_HEIGHT + padding;
    
    self.edgeImageView.frame = CGRectMake(leftX, leftY2, SAMPLE_IMAGE_WIDTH, SAMPLE_IMAGE_HEIGHT);
    self.backgroundImageView.frame = CGRectMake(leftX, leftY1, SAMPLE_IMAGE_WIDTH, SAMPLE_IMAGE_HEIGHT);
    
    self.rectImageView.frame = CGRectMake(0.0, 0.0, RECT_IMAGE_MIN_WIDTH, RECT_IMAGE_MIN_HEIGHT);
    self.rectImageView.frame = CGRectMake(0.0, 0.0, RECT_IMAGE_MIN_WIDTH, RECT_IMAGE_MIN_HEIGHT);
    
    self.debugLabel.frame = CGRectMake(0.0, self.view.frame.size.height - 30, self.view.frame.size.width, 30);
    NSLog(@"self.edgeImageView.layer.anchorPoint is: %@", NSStringFromCGPoint(self.edgeImageView.layer.anchorPoint));
    //2017-03-29 17:04:45.422 generate_training_dataset[15466:780934] self.edgeImageView.layer.anchorPoint is: {0.5, 0.5}  默认的锚点是中间点
}

- (CATransform3D)generateRandomTransform3DForXRotate {
    //http://www.jianshu.com/p/9cbf52eb39dd iOS动效-利用CATransform3D实现翻页动画效果
    //http://www.jianshu.com/p/32ac4d71cac7  CGAffineTransform  二维
    /**
     struct CATransform3D
     {
     CGFloat m11, m12, m13, m14;
     CGFloat m21, m22, m23, m24;
     CGFloat m31, m32, m33, m34;
     CGFloat m41, m42, m43, m44;
     };
     
     typedef struct CATransform3D CATransform3D;
     */
    

    CATransform3D transform = CATransform3DIdentity;//获取一个标准默认的CATransform3D仿射变换矩阵
    NSUInteger random_number = arc4random_uniform(41) + 60;//60 -- 100
    transform.m34 = -1.0 / random_number;//透视效果 http://stackoverflow.com/questions/10913676/why-does-a-calayers-transform3ds-m34-need-to-be-modified-before-applying-the-r
    
    
    
    //旋转+透视 !!不绕 Y 轴旋转，主要是要绕 X 轴宣传产生的透视效果
    CGFloat xRotate = arc4random_uniform(21) + 10;//X轴，10 -- 40度
    //CGFloat yRotate = arc4random_uniform(5) + 1 - 3;//Y轴，正负3度
    CGFloat zRotate = arc4random_uniform(20) + 1 - 10;//Z轴，小角度
    transform = CATransform3DRotate(transform, zRotate * M_PI / 360.0, 0, 0, 1);//绕 Z 轴，二维平面旋转
    transform = CATransform3DRotate(transform, xRotate * M_PI / 360.0, 1, 0, 0);//绕 X 轴
    //transform = CATransform3DRotate(transform, yRotate * M_PI / 360.0, 0, 1, 0);//绕 Y 轴，绕 X 和 Y 轴的旋转，都会产生透视效果
    
    
    //平移
    CGFloat xOffset = arc4random_uniform(SAMPLE_IMAGE_WIDTH - RECT_IMAGE_MIN_WIDTH) + 1;
    CGFloat yOffset = arc4random_uniform(SAMPLE_IMAGE_HEIGHT - RECT_IMAGE_MIN_HEIGHT) + 1;
    transform = CATransform3DTranslate(transform, xOffset, yOffset, 0);//x 和 y 轴平移，二维平面
    
    //缩放 缩放倍数 1.0 - 3.5 相当于
    CGFloat xScale = arc4random_uniform(25) + 10;//范围10 - 35
    CGFloat yScale = arc4random_uniform(25) + 10;
    transform = CATransform3DScale(transform, xScale * 0.1, yScale * 0.1, 0.0);
    
    return transform;
}

- (CATransform3D)generateRandomTransform3D {
    //http://www.jianshu.com/p/9cbf52eb39dd iOS动效-利用CATransform3D实现翻页动画效果
    //http://www.jianshu.com/p/32ac4d71cac7  CGAffineTransform  二维
    /**
     struct CATransform3D
     {
     CGFloat m11, m12, m13, m14;
     CGFloat m21, m22, m23, m24;
     CGFloat m31, m32, m33, m34;
     CGFloat m41, m42, m43, m44;
     };
     
     typedef struct CATransform3D CATransform3D;
     */
    
    CATransform3D transform = CATransform3DIdentity;//获取一个标准默认的CATransform3D仿射变换矩阵
    transform.m34 = -5.0 / 1000;//透视效果 http://stackoverflow.com/questions/10913676/why-does-a-calayers-transform3ds-m34-need-to-be-modified-before-applying-the-r
    
    
    
    //平移
    CGFloat xOffset = arc4random_uniform(SAMPLE_IMAGE_WIDTH - RECT_IMAGE_MIN_WIDTH) + 1;
    CGFloat yOffset = arc4random_uniform(SAMPLE_IMAGE_HEIGHT - RECT_IMAGE_MIN_HEIGHT) + 1;
    transform = CATransform3DTranslate(transform, xOffset, yOffset, 0);//x 和 y 轴平移，二维平面
    
    //缩放 缩放倍数 1.0 - 3.5 相当于
    CGFloat xScale = arc4random_uniform(25) + 10;//范围10 - 35
    CGFloat yScale = arc4random_uniform(25) + 10;
    transform = CATransform3DScale(transform, xScale * 0.1, yScale * 0.1, 0.0);
    
    //旋转+透视
    CGFloat xRotate = arc4random_uniform(90) + 1 - 45;//X和Y轴，最多旋转45度
    CGFloat yRotate = arc4random_uniform(45) + 1;
    CGFloat zRotate = arc4random_uniform(360) + 1;//Z轴，可以旋转1-360度
    transform = CATransform3DRotate(transform, xRotate * M_PI / 360, 1, 0, 0);//绕 X 轴
    transform = CATransform3DRotate(transform, yRotate * M_PI / 360, 0, 1, 0);//绕 Y 轴，绕 X 和 Y 轴的旋转，都会产生透视效果
    transform = CATransform3DRotate(transform, zRotate * M_PI / 360, 0, 0, 1);//绕 Z 轴，二维平面旋转
    
    return transform;
}

- (CATransform3D)generateRandomTransform3DForLargeImageVersion {
    CATransform3D transform = CATransform3DIdentity;//获取一个标准默认的CATransform3D仿射变换矩阵
    transform.m34 = -5.0 / 1000;
    
    //平移
    CGFloat xOffset = arc4random_uniform(50) + 1;
    CGFloat yOffset = arc4random_uniform(20) + 1;
    transform = CATransform3DTranslate(transform, xOffset, yOffset, 0);//x 和 y 轴平移，二维平面
    
    //缩放 缩放倍数 2.5 - 3.5
    CGFloat xScale = arc4random_uniform(10) + 25;//范围25 - 35
    CGFloat yScale = arc4random_uniform(10) + 25;
    transform = CATransform3DScale(transform, xScale * 0.1, yScale * 0.1, 0.0);
    
    //旋转+透视
    CGFloat xRotate = arc4random_uniform(10) + 1;//X和Y轴
    CGFloat yRotate = arc4random_uniform(10) + 1;
    CGFloat zRotate = arc4random_uniform(40) - 20;//Z轴
    transform = CATransform3DRotate(transform, xRotate * M_PI / 360, 1, 0, 0);//绕 X 轴
    transform = CATransform3DRotate(transform, yRotate * M_PI / 360, 0, 1, 0);//绕 Y 轴，绕 X 和 Y 轴的旋转，都会产生透视效果
    transform = CATransform3DRotate(transform, zRotate * M_PI / 360, 0, 0, 1);//绕 Z 轴，二维平面旋转
    
    return transform;
}

- (NSArray * _Nullable)getValidPointsWithTransform3D:(CATransform3D)transform {
    CGFloat leftTopX = 0.0;//debug的时候发现，如果 leftTopX 和 leftTopY 不为0，后面调用 convertPoint 得到的 point 的位置，和预期的还是有差别，所以，目前还是必须设置为 0
    CGFloat leftTopY = 0.0;//根据 viewWillLayoutSubviews 里面 self.rectImageView.frame 的值进行计算
    CGPoint leftTopPoint = CGPointMake(leftTopX, leftTopY);
    CGPoint rightTopPoint = CGPointMake(leftTopX + RECT_IMAGE_MIN_WIDTH, leftTopY);
    CGPoint rightBottomPoint = CGPointMake(leftTopX + RECT_IMAGE_MIN_WIDTH, leftTopY + RECT_IMAGE_MIN_HEIGHT);
    CGPoint leftBottomPoint = CGPointMake(leftTopX, leftTopY + RECT_IMAGE_MIN_HEIGHT);
    
    CGPoint p0, p1, p2, p3;
    //http://sketchytech.blogspot.com/2014/12/explaining-catransform3d-matrix.html
    p0 = [self.rectImageView.layer convertPoint:leftTopPoint toLayer:self.backgroundImageView.layer];
    p1 = [self.rectImageView.layer convertPoint:rightTopPoint toLayer:self.backgroundImageView.layer];
    p2 = [self.rectImageView.layer convertPoint:rightBottomPoint toLayer:self.backgroundImageView.layer];
    p3 = [self.rectImageView.layer convertPoint:leftBottomPoint toLayer:self.backgroundImageView.layer];
    /**
    NSLog(@"== p0 is: %@", NSStringFromCGPoint(p0));
    NSLog(@"== p1 is: %@", NSStringFromCGPoint(p1));
    NSLog(@"== p2 is: %@", NSStringFromCGPoint(p2));
    NSLog(@"== p3 is: %@", NSStringFromCGPoint(p3));
    */
    
    if (p0.x < 0.0 || p0.x > SAMPLE_IMAGE_WIDTH ||
        p0.y < 0.0 || p0.y > SAMPLE_IMAGE_HEIGHT ||
        p1.x < 0.0 || p1.x > SAMPLE_IMAGE_WIDTH ||
        p1.y < 0.0 || p1.y > SAMPLE_IMAGE_HEIGHT ||
        p2.x < 0.0 || p2.x > SAMPLE_IMAGE_WIDTH ||
        p2.y < 0.0 || p2.y > SAMPLE_IMAGE_HEIGHT ||
        p3.x < 0.0 || p3.x > SAMPLE_IMAGE_WIDTH ||
        p3.y < 0.0 || p3.y > SAMPLE_IMAGE_HEIGHT) {
        //任何一个 point，超出了 backgroundImageView 的范围，都返回 nil
        return nil;
    }
    
    NSArray *pointsArray = @[[NSValue valueWithCGPoint:p0],
                             [NSValue valueWithCGPoint:p1],
                             [NSValue valueWithCGPoint:p2],
                             [NSValue valueWithCGPoint:p3]];
    return pointsArray;
}

- (UIImage *)getCropBackgroundImage:(UIImage *)backgroundImage {
    NSInteger leftCrop = 0, topCrop = 0, rightCrop = 0, bottomCrop = 0;
    CGRect rect;
    
    const NSInteger MAX_CROP_LENGTH = 40;
    if (MIN(backgroundImage.size.width, backgroundImage.size.height) <= MAX_CROP_LENGTH * 2) {
        //如果backgroundImage最短的一条边的长度，小于 160 像素，则不进行裁剪抖动
        rect = CGRectMake(0.0, 0.0, backgroundImage.size.width, backgroundImage.size.height);
        NSLog(@"no need crop background image");
    } else {
        leftCrop = arc4random_uniform(MAX_CROP_LENGTH) + 1;
        topCrop = arc4random_uniform(MAX_CROP_LENGTH) + 1;
        rightCrop = arc4random_uniform(MAX_CROP_LENGTH) + 1;
        bottomCrop = arc4random_uniform(MAX_CROP_LENGTH) + 1;
        
        rect = CGRectMake(0.0 + leftCrop,
                          0.0 + topCrop,
                          backgroundImage.size.width - leftCrop - rightCrop,
                          backgroundImage.size.height - topCrop - bottomCrop);
    }
    
    backgroundImage = [backgroundImage croppedImageWithFrame:rect angle:0 circularClip:NO];
    return backgroundImage;
}


void draw1PxStroke(CGContextRef context, CGPoint startPoint, CGPoint endPoint, CGColorRef color) {
    CGContextSaveGState(context);
    CGContextSetLineCap(context, kCGLineCapSquare);
    CGContextSetStrokeColorWithColor(context, color);
    //CGContextSetLineWidth(context, 1.0);
    CGContextSetLineWidth(context, 0.2);//这里虽然是用白色进行绘制，但是是有平滑处理的，所以有些白色的点对应的值，并不是255。在用 python 对这里得到的图像做进一步处理的时候，还调用了cv2.threshold(grayImage, 20, 255, cv2.THRESH_BINARY)进一步做二值化处理
    CGContextMoveToPoint(context, startPoint.x + 0.5, startPoint.y + 0.5);
    CGContextAddLineToPoint(context, endPoint.x + 0.5, endPoint.y + 0.5);
    CGContextStrokePath(context);
    CGContextRestoreGState(context);
}


- (UIImage *)drawPoints:(NSArray *)points {
    NSParameterAssert(points != nil);
    NSParameterAssert(points.count == 4);
    
    NSValue *pointValue = points[0];
    CGPoint p0 = [pointValue CGPointValue];
    
    pointValue = points[1];
    CGPoint p1 = [pointValue CGPointValue];
    
    pointValue = points[2];
    CGPoint p2 = [pointValue CGPointValue];
    
    pointValue = points[3];
    CGPoint p3 = [pointValue CGPointValue];
    //NSLog(@"###### p are: %@, %@, %@, %@", NSStringFromCGPoint(p0), NSStringFromCGPoint(p1), NSStringFromCGPoint(p2), NSStringFromCGPoint(p3));
    
    
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(SAMPLE_IMAGE_WIDTH, SAMPLE_IMAGE_HEIGHT), NO, [UIScreen mainScreen].scale);
    CGContextRef context = UIGraphicsGetCurrentContext();
    
    draw1PxStroke(context, p0, p1, [UIColor whiteColor].CGColor);
    draw1PxStroke(context, p1, p2, [UIColor whiteColor].CGColor);
    draw1PxStroke(context, p2, p3, [UIColor whiteColor].CGColor);
    draw1PxStroke(context, p3, p0, [UIColor whiteColor].CGColor);
    
    UIImage *resultImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return resultImage;
}



- (void)writeImageToFileWithNamePrefix:(NSString *)prefix {
    UIImage *colorImage = [UIImage imageWithView_version2:self.backgroundImageView];
    UIImage *grayImage = [UIImage imageWithView:self.edgeImageView];//edgeImageView上是没有CATransform3D的，所以截图正常
    
    NSString *colorImagePath = [self.imageSaveFolder stringByAppendingPathComponent:[NSString stringWithFormat:@"%@_color.jpg", prefix]];
    [UIImageJPEGRepresentation(colorImage, 1.0) writeToFile:colorImagePath atomically:YES];
    
    NSString *grayImagePath = [self.imageSaveFolder stringByAppendingPathComponent:[NSString stringWithFormat:@"%@_annotation.png", prefix]];
    [UIImagePNGRepresentation(grayImage) writeToFile:grayImagePath atomically:YES];
}



- (void)generateShadowForRectImageView {
    //TODO, 增加 shadow 后，出现了诡异的 bug，rectImageView 的 frame 会奇怪的错乱一部分，有一定的概率，很容易重现出来，暂时先不增加 shadow 了
    
    /**
    CGFloat offsetX = arc4random_uniform(9) - 4.0;
    CGFloat offsetY = arc4random_uniform(9) - 4.0;
    CGFloat opacity = arc4random_uniform(5) / 10.0 + 0.5;
    CGFloat radius = arc4random_uniform(4) + 1;

    //不能在self.interferentImageView上模拟阴影，会出现不想要的效果
    //NSLog(@"(%f, %f), %f, %f", offsetX, offsetY, opacity, radius);
    self.rectImageView.layer.shadowColor = [UIColor blackColor].CGColor;
    self.rectImageView.layer.shadowOffset = CGSizeMake(offsetX, offsetY);
    self.rectImageView.layer.shadowOpacity = opacity;
    self.rectImageView.layer.shadowRadius = radius;
    */
}

- (void)generateDataset {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];//生成的 image 保存到这个目录里
    NSFileManager *fileManager = [NSFileManager defaultManager];
    
    
    NSUInteger totalBackgroundImageCount = self.backgroundImageFileNames.count;
    NSUInteger currentBackgroundImageIndex = 0;
    NSUInteger totalRectImageCount = self.rectImageFileNames.count;
    NSUInteger currentRectImageIndex = 0;
    NSUInteger __block totalTransformCount = 0;
    NSUInteger __block totalValidTransformCount = 0;
    
    for (NSString *backgroundImageName in self.backgroundImageFileNames) {
        UIImage *backgroundImage = [UIImage imageWithContentsOfFile:[self.backgroundImagesPath stringByAppendingPathComponent:backgroundImageName]];
        if (backgroundImage == nil) {
            NSLog(@"background image is bad: %@", backgroundImageName);
            continue;
        }
        
        currentRectImageIndex = 0;
        
        for (NSString *rectImageName in self.rectImageFileNames) {
            UIImage *rectImage = [UIImage imageWithContentsOfFile:[self.rectImagesPath stringByAppendingPathComponent:rectImageName]];
            if (rectImage == nil) {
                NSLog(@"rect image is bad: %@", rectImageName);
                continue;
            }
            
            NSInteger __block validTransformCount = 0;
            
            //1张随机尺寸的图
            while (validTransformCount < 1) {
                dispatch_sync(dispatch_get_main_queue(), ^{
                    @autoreleasepool {
                        CATransform3D transform = [self generateRandomTransform3D];
                        totalTransformCount++;
                        
                        UIImage *cropedBackgroundImage = [self getCropBackgroundImage:backgroundImage];
                        self.backgroundImageView.image = cropedBackgroundImage;
                        
                        self.rectImageView.image = rectImage;
                        [self generateShadowForRectImageView];
                        
                        self.rectImageView.layer.anchorPoint = CGPointZero;//这个锚点设置很重要，还没有搞清楚 anchorPoint 对 transform 的影响
                        self.rectImageView.layer.transform = transform;
                        //!!!!!!一定要先调用 self.rectImageView.layer.transform = transform 这行代码，然后 getValidPointsWithTransform3D 里面的 convertPoint 才会做出正确的计算
                        NSArray *points = [self getValidPointsWithTransform3D:transform];
                        
                        if (points) {
                            validTransformCount++;
                            totalValidTransformCount++;
                            
                            self.edgeImageView.image = [self drawPoints:points];
                            
                            NSString *prefix = [NSString stringWithFormat:@"%@_random_size_%lu_%lu_%lu", [NSString randomStringWithLength:10], currentBackgroundImageIndex, currentRectImageIndex, validTransformCount];
                            [self writeImageToFileWithNamePrefix:prefix];
                            NSLog(@"--debug, %lu-%lu, %@, %@", currentBackgroundImageIndex, currentRectImageIndex, backgroundImageName, rectImageName);
                        } else {
                            //NSLog(@"%s, get invalid transform and point, just pass", __PRETTY_FUNCTION__);
                        }
                    }
                });
            }
            
            //1张大尺寸的图
            validTransformCount = 0;
            while (validTransformCount < 1) {
                dispatch_sync(dispatch_get_main_queue(), ^{
                    @autoreleasepool {
                        CATransform3D transform = [self generateRandomTransform3DForLargeImageVersion];
                        totalTransformCount++;
                        
                        UIImage *cropedBackgroundImage = [self getCropBackgroundImage:backgroundImage];
                        self.backgroundImageView.image = cropedBackgroundImage;
                        
                        self.rectImageView.image = rectImage;
                        [self generateShadowForRectImageView];
                        
                        self.rectImageView.layer.anchorPoint = CGPointZero;//这个锚点设置很重要，还没有搞清楚 anchorPoint 对 transform 的影响
                        self.rectImageView.layer.transform = transform;
                        //!!!!!!一定要先调用 self.rectImageView.layer.transform = transform 这行代码，然后 getValidPointsWithTransform3D 里面的 convertPoint 才会做出正确的计算
                        NSArray *points = [self getValidPointsWithTransform3D:transform];
                        
                        if (points) {
                            validTransformCount++;
                            totalValidTransformCount++;
                            
                            self.edgeImageView.image = [self drawPoints:points];
                            
                            NSString *prefix = [NSString stringWithFormat:@"%@_large_size_%lu_%lu_%lu", [NSString randomStringWithLength:10], currentBackgroundImageIndex, currentRectImageIndex, validTransformCount];
                            [self writeImageToFileWithNamePrefix:prefix];
                            
                        } else {
                            //NSLog(@"%s, get invalid transform and point, just pass", __PRETTY_FUNCTION__);
                        }
                    }
                });
            }
            
            //1张绕X轴旋转的图
            validTransformCount = 0;
            while (validTransformCount < 1) {
                dispatch_sync(dispatch_get_main_queue(), ^{
                    @autoreleasepool {
                        CATransform3D transform = [self generateRandomTransform3DForXRotate];
                        totalTransformCount++;
                        
                        UIImage *cropedBackgroundImage = [self getCropBackgroundImage:backgroundImage];
                        self.backgroundImageView.image = cropedBackgroundImage;
                        
                        self.rectImageView.image = rectImage;
                        [self generateShadowForRectImageView];
                        
                        self.rectImageView.layer.anchorPoint = CGPointZero;//这个锚点设置很重要，还没有搞清楚 anchorPoint 对 transform 的影响
                        self.rectImageView.layer.transform = transform;
                        //!!!!!!一定要先调用 self.rectImageView.layer.transform = transform 这行代码，然后 getValidPointsWithTransform3D 里面的 convertPoint 才会做出正确的计算
                        NSArray *points = [self getValidPointsWithTransform3D:transform];
                        
                        if (points) {
                            validTransformCount++;
                            totalValidTransformCount++;
                            
                            self.edgeImageView.image = [self drawPoints:points];
                            
                            NSString *prefix = [NSString stringWithFormat:@"%@_large_size_%lu_%lu_%lu", [NSString randomStringWithLength:10], currentBackgroundImageIndex, currentRectImageIndex, validTransformCount];
                            [self writeImageToFileWithNamePrefix:prefix];
                            
                        } else {
                            //NSLog(@"%s, get invalid transform and point, just pass", __PRETTY_FUNCTION__);
                        }
                    }
                });
            }
            
            currentRectImageIndex++;
            

            //update debug info
            NSString *fullDebugInfo = [NSString stringWithFormat:@"background: %lu/%lu, rect: %lu/%lu, valid transform: %lu, invalid transform: %lu", (unsigned long)currentBackgroundImageIndex, (unsigned long)totalBackgroundImageCount, (unsigned long)currentRectImageIndex, (unsigned long)totalRectImageCount, (unsigned long)totalValidTransformCount, totalTransformCount - totalValidTransformCount];
            NSString *debugInfo = [NSString stringWithFormat:@"B: %lu/%lu, R: %lu/%lu, valid: %lu, invalid: %lu", (unsigned long)currentBackgroundImageIndex, (unsigned long)totalBackgroundImageCount, (unsigned long)currentRectImageIndex, (unsigned long)totalRectImageCount, (unsigned long)totalValidTransformCount, totalTransformCount - totalValidTransformCount];
            
            dispatch_sync(dispatch_get_main_queue(), ^{
                self.debugLabel.text = debugInfo;
            });
            
            NSLog(@"%@", fullDebugInfo);
        }
        
        currentBackgroundImageIndex++;
    }
}
@end






@implementation UIImage (Utilities)

+ (UIImage *)imageWithView:(UIView *)view {
    /** 这个版本的截图，不会带有 CATransform3D 的效果
     transform = CATransform3DIdentity;//获取一个标准默认的CATransform3D仿射变换矩阵
     transform.m34 = -5.0 / 1000;//透视效果 http://stackoverflow.com/questions/10913676/why-does-a-calayers-transform3ds-m34-need-to-be-modified-before-applying-the-r
     transform = CATransform3DRotate(transform, 20 * M_PI / 360, 0, 0, 1);//绕 Z 轴，二维平面旋转
     self.edgeImageView.layer.transform = transform;
     
     //uiview 上的 CGAffineTransform 变换，可以被截图，但是 CALayer 上的 CATransform3D，不能被截图
     float degrees = 20; //the value in degrees
     self.edgeImageView.transform = CGAffineTransformMakeRotation(degrees * M_PI/180);
     
     UIImage *grayImage = [UIImage imageWithView:self.view];
     */
    
    //UIGraphicsBeginImageContextWithOptions(view.bounds.size, view.opaque, [UIScreen mainScreen].scale);
    UIGraphicsBeginImageContextWithOptions(view.bounds.size, view.opaque, 1.0);
    [view.layer renderInContext:UIGraphicsGetCurrentContext()];
    
    UIImage *img = UIGraphicsGetImageFromCurrentImageContext();
    
    UIGraphicsEndImageContext();
    
    return img;
}

+ (UIImage *)imageWithView_version2:(UIView *)view {
    //http://stackoverflow.com/questions/31453906/capturing-screenshot-of-uiview-with-multiple-calayers-displays-white-background
    //用drawViewHierarchyInRect截图，才能有CATransform3D的效果
    UIGraphicsBeginImageContextWithOptions(view.bounds.size, NO, 1.0);
    [view drawViewHierarchyInRect:view.bounds afterScreenUpdates:YES];
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}

//http://stackoverflow.com/questions/2658738/the-simplest-way-to-resize-an-uiimage
- (UIImage *)scaleToSize:(CGSize)size {
    UIGraphicsBeginImageContextWithOptions(size, NO, self.scale);
    [self drawInRect:CGRectMake(0, 0, size.width, size.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

//http://stackoverflow.com/questions/158914/cropping-an-uiimage
- (UIImage *)crop:(CGRect)rect {
    if (self.scale > 1.0f) {
        rect = CGRectMake(rect.origin.x * self.scale,
                          rect.origin.y * self.scale,
                          rect.size.width * self.scale,
                          rect.size.height * self.scale);
    }
    
    CGImageRef imageRef = CGImageCreateWithImageInRect(self.CGImage, rect);
    UIImage *result = [UIImage imageWithCGImage:imageRef scale:self.scale orientation:self.imageOrientation];
    CGImageRelease(imageRef);
    return result;
}

- (BOOL)hasAlpha {
    CGImageAlphaInfo alphaInfo = CGImageGetAlphaInfo(self.CGImage);
    return (alphaInfo == kCGImageAlphaFirst || alphaInfo == kCGImageAlphaLast ||
            alphaInfo == kCGImageAlphaPremultipliedFirst || alphaInfo == kCGImageAlphaPremultipliedLast);
}

- (UIImage *)croppedImageWithFrame:(CGRect)frame angle:(NSInteger)angle circularClip:(BOOL)circular {
    UIImage *croppedImage = nil;
    UIGraphicsBeginImageContextWithOptions(frame.size, ![self hasAlpha] && !circular, self.scale);
    {
        CGContextRef context = UIGraphicsGetCurrentContext();
        
        if (circular) {
            CGContextAddEllipseInRect(context, (CGRect){CGPointZero, frame.size});
            CGContextClip(context);
        }
        
        //To conserve memory in not needing to completely re-render the image re-rotated,
        //map the image to a view and then use Core Animation to manipulate its rotation
        if (angle != 0) {
            UIImageView *imageView = [[UIImageView alloc] initWithImage:self];
            imageView.layer.minificationFilter = kCAFilterNearest;
            imageView.layer.magnificationFilter = kCAFilterNearest;
            imageView.transform = CGAffineTransformRotate(CGAffineTransformIdentity, angle * (M_PI/180.0f));
            CGRect rotatedRect = CGRectApplyAffineTransform(imageView.bounds, imageView.transform);
            UIView *containerView = [[UIView alloc] initWithFrame:(CGRect){CGPointZero, rotatedRect.size}];
            [containerView addSubview:imageView];
            imageView.center = containerView.center;
            CGContextTranslateCTM(context, -frame.origin.x, -frame.origin.y);
            [containerView.layer renderInContext:context];
        }
        else {
            CGContextTranslateCTM(context, -frame.origin.x, -frame.origin.y);
            [self drawAtPoint:CGPointZero];
        }
        
        croppedImage = UIGraphicsGetImageFromCurrentImageContext();
    }
    UIGraphicsEndImageContext();
    
    return [UIImage imageWithCGImage:croppedImage.CGImage scale:[UIScreen mainScreen].scale orientation:UIImageOrientationUp];
}
@end









@implementation NSString (Utilities)
NSString const *letters = @"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

+ (NSString *)randomStringWithLength:(int)length {
    NSMutableString *randomString = [NSMutableString stringWithCapacity:length];
    
    for (int i=0; i<length; i++) {
        [randomString appendFormat: @"%C", [letters characterAtIndex: arc4random_uniform((uint32_t)[letters length])]];
    }
    
    return randomString;
}
@end

//////////////////////////////////
