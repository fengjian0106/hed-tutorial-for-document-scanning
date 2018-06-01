//
//  fm_ocr_scanner.cpp
//  FMHEDNet
//
//  Created by fengjian on 2018/4/11.
//  Copyright © 2018年 fengjian. All rights reserved.
//

#include "fm_ocr_scanner.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//在具体调用的地方，看这几个常量的解释
const int kHoughLinesPThreshold = 20;
const double kHoughLinesPMinLinLength = 20.0;
const double kHoughLinesPMaxLineGap = 3.0;

const int kMergeLinesMaxDistance = 5;
const int kIntersectionMinAngle = 45;
const int kIntersectionMaxAngle = 135;
const double kCloserPointMaxDistance = 6.0;
const double kRectOpposingSidesMinRatio = 0.5;
const int kPointOnLineMaxOffset = 8;
const int kSameSegmentsMaxAngle = 5;


struct Corner {
    cv::Point point;
    std::vector<cv::Vec4i> segments;
};


static bool IsPointOnLine(const cv::Point point, const cv::Vec4i line) {
    cv::Point p0 = cv::Point(line[0], line[1]);
    cv::Point p1 = cv::Point(line[2], line[3]);
    
    int min_x, max_x, min_y, max_y;
    min_x = MIN(p0.x, p1.x) - kPointOnLineMaxOffset;//在HED和霍夫曼检测的时候，矩形的拐角处的两条线段，可能会断开，所以这里在line的两端，适当的延长一点点距离
    max_x = MAX(p0.x, p1.x) + kPointOnLineMaxOffset;
    min_y = MIN(p0.y, p1.y) - kPointOnLineMaxOffset;
    max_y = MAX(p0.y, p1.y) + kPointOnLineMaxOffset;
    
    if (point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y) {
        return true;
    }
    
    return false;
}

//https://gist.github.com/ceykmc/18d3f82aaa174098f145
static std::array<int, 3> Cross(const std::array<int, 3> &a,
                                const std::array<int, 3> &b) {
    std::array<int, 3> result;
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

//这个版本，line 是看成一条可以无限延长的直线
static bool GetIntersection(const cv::Vec4i &line_a, const cv::Vec4i &line_b, cv::Point &intersection) {
    std::array<int, 3> pa{ { line_a[0], line_a[1], 1 } };
    std::array<int, 3> pb{ { line_a[2], line_a[3], 1 } };
    std::array<int, 3> la = Cross(pa, pb);
    pa[0] = line_b[0], pa[1] = line_b[1], pa[2] = 1;
    pb[0] = line_b[2], pb[1] = line_b[3], pb[2] = 1;
    std::array<int, 3> lb = Cross(pa, pb);
    std::array<int, 3> inter = Cross(la, lb);
    if (inter[2] == 0) return false; // two lines are parallel
    else {
        intersection.x = inter[0] / inter[2];
        intersection.y = inter[1] / inter[2];
        return true;
    }
}

//这个版本，line实际上是有限长度的线段，所以还额外检测了一下 point 是否在线段上
static bool GetSegmentIntersection(const cv::Vec4i &line_a, const cv::Vec4i &line_b, cv::Point &intersection) {
    std::array<int, 3> pa{ { line_a[0], line_a[1], 1 } };
    std::array<int, 3> pb{ { line_a[2], line_a[3], 1 } };
    std::array<int, 3> la = Cross(pa, pb);
    
    pa[0] = line_b[0];
    pa[1] = line_b[1];
    pa[2] = 1;
    
    pb[0] = line_b[2];
    pb[1] = line_b[3];
    pb[2] = 1;
    
    std::array<int, 3> lb = Cross(pa, pb);
    std::array<int, 3> inter = Cross(la, lb);
    if (inter[2] == 0) return false; // two lines are parallel
    else {
        intersection.x = inter[0] / inter[2];
        intersection.y = inter[1] / inter[2];
        
        if (IsPointOnLine(intersection, line_a) == true && IsPointOnLine(intersection, line_b) == true) {
            return true;
        }
        
        return false;
    }
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/**
 atan2的返回值，并不是0~360度，而是0~180度 和/或 0~-180度，之前就是掉到这个坑里了，所以需要修正一下
 
 收集的一些真实数据，注意第二行的  (-176.228, 173.9)，一个正一个负，数值其实是正确的，因为两条线都接近于水平线，一个向上倾斜一个向下倾斜，也就是一个小于180度，一个大于180度，在数学上是等价于用一个正角度表示小于180的，用一个负角度表示大于180的那个角。
 ------- debug, (angle_top, angle_bottom) are: (178.652, 179.599), (angle_right, angle_left) are: (-101.136, -75.3236)
 ------- debug, (angle_top, angle_bottom) are: (-176.228, 173.9), (angle_right, angle_left) are: (-114.411, -97.219)
 ------- debug, (angle_top, angle_bottom) are: (-142.927, -157.126), (angle_right, angle_left) are: (-62.549, -54.9165)
 ------- debug, (angle_top, angle_bottom) are: (-176.576, -179.441), (angle_right, angle_left) are: (-107.324, -64.6538)
 */
/**
 下面这两个是未修正版本，之前就是在这里引出了 bug
 static double GetAngleOfLine(const cv::Vec4i &line) {
 int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
 
 double angle = atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
 
 return angle;
 }
 
 static double GetAngleOfTwoPoints(const cv::Point &point_a, const cv::Point &point_b) {
 double angle = atan2(point_b.y - point_a.y, point_b.x - point_a.x) * 180.0 / CV_PI;
 return angle;
 }
 */

//http://opencv-users.1802565.n2.nabble.com/Angle-between-2-lines-td6803229.html
//http://stackoverflow.com/questions/2339487/calculate-angle-of-2-points
static int GetAngleOfLine(const cv::Vec4i &line) {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
    
    //http://stackoverflow.com/questions/1311049/how-to-map-atan2-to-degrees-0-360
    //degrees = (degrees + 360) % 360 这种修正办法，得到的是 int 类型的角度，虽然损失了一点点精度，但是还是可以满足这里算法的需求
    double angle = atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
    int fix_angle = ((int)angle + 360) % 360;
    
    assert(fix_angle >= 0);
    assert(fix_angle <= 360);
    return fix_angle;
}

static int GetAngleOfTwoPoints(const cv::Point &point_a, const cv::Point &point_b) {
    double angle = atan2(point_b.y - point_a.y, point_b.x - point_a.x) * 180.0 / CV_PI;
    int fix_angle = ((int)angle + 360) % 360;
    
    assert(fix_angle >= 0);
    assert(fix_angle <= 360);
    return fix_angle;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/**
 RefLineVec4i 比较特殊，如果把它看成向量的话，它的反向是遵守一定的规则的。
 RefLineVec4i 里面的两个点，总是从左往右的方向，如果 RefLine 和 Y 轴平行(区分不了左右)，则按照从下往上的方向
 */
typedef cv::Vec4i RefLineVec4i;

static bool IsTwoRefLineCloseToEachOther(RefLineVec4i line_a, RefLineVec4i line_b) {
    if (std::abs(line_a[1] - line_b[1]) < kMergeLinesMaxDistance && std::abs(line_a[3] - line_b[3]) < kMergeLinesMaxDistance) {
        return true;
    }
    
    return false;
}


static RefLineVec4i GetRefLine(const cv::Vec4i line, int image_width, int image_height) {
    /**
     line format is (x_{start}, y_{start}, x_{end}, y_{end})
     分别对应          line[0]    line[1]    line[2]  line[3]
     
     公式为 y = a*x + b
     
     根据下面两个等式
     line[1] = a*line[0] + b
     line[3] = a*line[2] + b
     
     可以进行推导
     line[1] - line[3] = a* (line[0] - line[2])
     
     得到
     a = (line[1] - line[3]) / (line[0] - line[2])
     b = line[1] - a*line[0]
       = (line[0]*line[3] - line[2]*line[1]) / (line[0] - line[2])
     */
    
    RefLineVec4i ref_line;
    
    if (line[0] == line[2]) {
        //和 Y 轴平行的线，按照从下往上的方向排列
        ref_line[0] = line[0];
        ref_line[1] = 0; //从下往上
        ref_line[2] = line[2];
        ref_line[3] = image_height;
    } else if (line[1] == line[3]) {
        //和 X 轴平行的线，按照从左往右的方向
        ref_line[0] = 0; //从左往右
        ref_line[1] = line[1];
        ref_line[2] = image_width;
        ref_line[3] = line[3];
    } else {
        //这个分支的斜线才能通过公式进行计算，而且不会遇到下面这个除法中 (line[0] - line[2]) == 0 的情况，避免计算错误
        //a = (line[1] - line[3]) / (line[0] - line[2])
        
        float a, b;
        a = (float)(line[1] - line[3]) / (float)(line[0] - line[2]);
        b = (float)(line[0]*line[3] - line[2]*line[1]) / (float)(line[0] - line[2]);
        
        // y = a*x + b
        ref_line[0] = 0; //从左往右
        ref_line[1] = int(b);
        ref_line[2] = int((image_height - b) / a);
        ref_line[3] = image_height;// ref_line[3] = a*ref_line[2] + b
        
        //std::cout << "__ ref_line are: (" << ref_line[0] << ", " << ref_line[1] << ", " << ref_line[2] << ", " << ref_line[3] << ")" << std::endl;
    }
 
    return ref_line;
}

static bool SortPointsByXaxis(const cv::Point &a, const cv::Point &b) {
    return a.x < b.x;
}

static bool SortPointsByYaxis(const cv::Point &a, const cv::Point &b) {
    return a.y < b.y;
}

static bool SortCornersByXaxis(const Corner &a, const Corner &b) {
    return a.point.x < b.point.x;
}

static bool SortCornersByYaxis(const Corner &a, const Corner &b) {
    return a.point.y < b.point.y;
}

static bool IsSegmentsHasSameSegment(const std::vector<cv::Vec4i> segments, const cv::Vec4i segment, int image_width) {
    for (int i = 0; i < segments.size(); i++) {
        cv::Vec4i seg = segments[i];
        
        int angle_a = GetAngleOfLine(seg);
        int angle_b = GetAngleOfLine(segment);
        
        int diff = std::abs(angle_a - angle_b);
        diff = diff % 90;//修正到0~90度
        
        //std::cout << " ********************, angle_a, angle_b are: (" << angle_a << ", " << angle_b << "), diff is: " << diff << std::endl;
        if (diff < kSameSegmentsMaxAngle) {
            return true;
        }
    }
    
    //TODO，还可以考虑是否需要更严格的判断策略
    return false;
}

/**
 HoughLinesP检测出来的是线段，是有长度的。
 把每一个线段扩展成一个统一规格的 RefLineVec4i，形成一个 pair，然后调用这个函数，对这些 pair 进行合并。
 用RefLineVec4i来决策是否需要合并，对于需要合并的，则把对应的HoughLinesP线段重组成一个更长的线段。
 */
static std::vector<std::tuple<RefLineVec4i, cv::Vec4i> > MergeRefLineAndSegmentPairs(std::vector<std::tuple<RefLineVec4i, cv::Vec4i> > ref_line_and_segment_pairs, int image_width, int image_height) {
    std::vector<std::tuple<RefLineVec4i, cv::Vec4i> > merged_ref_line_and_segment_pairs;
    
    for (int i = 0; i < ref_line_and_segment_pairs.size(); i++) {
        std::tuple<RefLineVec4i, cv::Vec4i> ref_line_and_segment = ref_line_and_segment_pairs[i];
        
        auto ref_line = std::get<0>(ref_line_and_segment);
        auto segment = std::get<1>(ref_line_and_segment);
        
        if (merged_ref_line_and_segment_pairs.size() == 0) {
            merged_ref_line_and_segment_pairs.push_back(std::make_tuple(ref_line, segment));
        } else {
            bool isCloser = false;
            for (int j = 0; j < merged_ref_line_and_segment_pairs.size(); j++) {
                auto merged_ref_line_and_segment = merged_ref_line_and_segment_pairs[j];
                auto merged_ref_line = std::get<0>(merged_ref_line_and_segment);
                auto merged_segment = std::get<1>(merged_ref_line_and_segment);
                
                //std::cout << "debug, std::abs(line[1] - merged_line[1]) " << std::abs(line[1] - merged_line[1]) << ", std::abs(line[3] - merged_line[3]) " << std::abs(line[3] - merged_line[3]) << std::endl;
                
                if (IsTwoRefLineCloseToEachOther(ref_line, merged_ref_line) == true) {
                    //如果两条 ref line 很接近，则把两个segment合并成一个，然后重新生成新的 ref line
                    
                    //先取出4个点
                    cv::Point p0 = cv::Point(segment[0], segment[1]);
                    cv::Point p1 = cv::Point(segment[2], segment[3]);
                    cv::Point p2 = cv::Point(merged_segment[0], merged_segment[1]);
                    cv::Point p3 = cv::Point(merged_segment[2], merged_segment[3]);
                    
                    std::vector<cv::Point> point_vector;
                    point_vector.push_back(p0);
                    point_vector.push_back(p1);
                    point_vector.push_back(p2);
                    point_vector.push_back(p3);
                    
                    //排序之后，得到最左边和最右边的两个 point，这两个 point 就可以构成新的线段
                    std::sort(point_vector.begin(), point_vector.end(), SortPointsByXaxis);
                    cv::Point left_most_point = point_vector[0];
                    cv::Point right_most_point = point_vector[3];
                    
                    cv::Vec4i new_segment;
                    new_segment[0] = left_most_point.x;
                    new_segment[1] = left_most_point.y;
                    new_segment[2] = right_most_point.x;
                    new_segment[3] = right_most_point.y;
                    //TODO，考虑一下，这里是否需要使用其他的线段合并策略，是否需要把新的线段的两个 point，做一个细微调整，让这两个 point 正好处于新的直线上
                    
                    RefLineVec4i new_ref_line = GetRefLine(new_segment, image_width, image_height);
                    merged_ref_line_and_segment_pairs[j] = std::make_tuple(new_ref_line, new_segment);
                    isCloser = true;
                    break;
                }
            }
            
            if (isCloser == false) {
                merged_ref_line_and_segment_pairs.push_back(std::make_tuple(ref_line, segment));
            }
        }
    }
    
    return merged_ref_line_and_segment_pairs;
}

static double PointsDistance(const cv::Point &a, const cv::Point &b) {
    double x_distance = (double)a.x - (double)b.x;
    double y_distance = (double)a.y - (double)b.y;
    
    double distance = cv::sqrt(x_distance * x_distance + y_distance * y_distance);
    
    //std::cout << " -- pointsDistance, [x_distance, y_distance, distance] are: [" << x_distance << ", " << y_distance << ", " << distance << "]" << std::endl;
    return distance;
}

/**
 按照顺时针排序，对4个 corner 排序，得到 4 corners: top-left, top-right, bottom-right, bottom-left, index are 0, 1, 2, 3
 */
static std::vector<Corner> ArrangeRectCorners(std::vector<Corner> rect_corners) {
    assert(rect_corners.size() == 4);
    
    std::sort(rect_corners.begin(), rect_corners.end(), SortCornersByXaxis);
    
    std::vector<Corner> left_two_corners;
    std::vector<Corner> right_two_corners;
    left_two_corners.push_back(rect_corners[0]);
    left_two_corners.push_back(rect_corners[1]);
    right_two_corners.push_back(rect_corners[2]);
    right_two_corners.push_back(rect_corners[3]);
    
    std::sort(left_two_corners.begin(), left_two_corners.end(), SortCornersByYaxis);
    std::sort(right_two_corners.begin(), right_two_corners.end(), SortCornersByYaxis);
    
    std::vector<Corner> sorted_corners;// top-left, top-right, bottom-right, bottom-left
    sorted_corners.push_back(left_two_corners[0]);
    sorted_corners.push_back(right_two_corners[0]);
    sorted_corners.push_back(right_two_corners[1]);
    sorted_corners.push_back(left_two_corners[1]);
    
    return sorted_corners;
}

/**
 一组策略，判断4个 corner 是否可以形成一个可信度高的矩形(有透视变换效果，所以肯定不是标准的长方形，而是一个梯形或平行四边形)
 4个 point 是已经经过ArrangeRectPoints排过序的
 4 points top-left, top-right, bottom-right, bottom-left, index are 0, 1, 2, 3
 */
static bool IsRectCornersReasonable(std::vector<Corner> rect_corners, int image_width) {
    assert(rect_corners.size() == 4);
    
    //第一组策略，根据之前记录的 segment 和四边形每条边的相似度进行过滤
    std::vector<cv::Point> rect_points;
    rect_points.push_back(rect_corners[0].point);
    rect_points.push_back(rect_corners[1].point);
    rect_points.push_back(rect_corners[2].point);
    rect_points.push_back(rect_corners[3].point);
    
    cv::Vec4i segment_0_to_1 = cv::Vec4i(rect_points[0].x, rect_points[0].y, rect_points[1].x, rect_points[1].y);
    cv::Vec4i segment_1_to_2 = cv::Vec4i(rect_points[1].x, rect_points[1].y, rect_points[2].x, rect_points[2].y);
    cv::Vec4i segment_2_to_3 = cv::Vec4i(rect_points[2].x, rect_points[2].y, rect_points[3].x, rect_points[3].y);
    cv::Vec4i segment_3_to_0 = cv::Vec4i(rect_points[3].x, rect_points[3].y, rect_points[0].x, rect_points[0].y);
    
    std::vector<cv::Vec4i> rect_segments;
    rect_segments.push_back(segment_0_to_1);
    rect_segments.push_back(segment_1_to_2);
    rect_segments.push_back(segment_2_to_3);
    rect_segments.push_back(segment_3_to_0);
    
    
    /**
     segment_0_to_1这条线段，应该和rect_corners[0]的所有 segments 里面的至少一条线段是相似的，同时，
     segment_0_to_1这条线段，也应该和rect_corners[1]的所有 segments 里面的至少一条线段是相似的
     */
    if (IsSegmentsHasSameSegment(rect_corners[0].segments, segment_0_to_1, image_width) &&
        IsSegmentsHasSameSegment(rect_corners[1].segments, segment_0_to_1, image_width)) {
        
    } else {
        return false;
    }
    
    if (IsSegmentsHasSameSegment(rect_corners[1].segments, segment_1_to_2, image_width) &&
        IsSegmentsHasSameSegment(rect_corners[2].segments, segment_1_to_2, image_width)) {
        
    } else {
        return false;
    }
    
    if (IsSegmentsHasSameSegment(rect_corners[2].segments, segment_2_to_3, image_width) &&
        IsSegmentsHasSameSegment(rect_corners[3].segments, segment_2_to_3, image_width)) {
        
    } else {
        return false;
    }
    
    if (IsSegmentsHasSameSegment(rect_corners[3].segments, segment_3_to_0, image_width) &&
        IsSegmentsHasSameSegment(rect_corners[0].segments, segment_3_to_0, image_width)) {
        
    } else {
        return false;
    }
    
    
    //第二组策略，根据四边形的形状
    double distance_of_0_to_1 = PointsDistance(rect_points[0], rect_points[1]);
    double distance_of_1_to_2 = PointsDistance(rect_points[1], rect_points[2]);
    double distance_of_2_to_3 = PointsDistance(rect_points[2], rect_points[3]);
    double distance_of_3_to_0 = PointsDistance(rect_points[3], rect_points[0]);
    
    
    //计算两组对边的比例(0.0 -- 1.0的值)
    //两条对边(标准矩形的时候，就是两条平行边)的 minLength / maxLength，不能小于0.5，否则就认为不是矩形(本来是把这个阈值设置为0.8的，但是因为图片都是缩放后进行的处理，长宽比有很大的变化，所以把这里的过滤条件放宽一些，设置为0.5)
    //distance_of_0_to_1 和 distance_of_2_to_3 是两条对边
    double ratio1 = MIN(distance_of_0_to_1, distance_of_2_to_3) / MAX(distance_of_0_to_1, distance_of_2_to_3);
    double ratio2 = MIN(distance_of_1_to_2, distance_of_3_to_0) / MAX(distance_of_1_to_2, distance_of_3_to_0);
    
    //std::cout << " ------- debug, distance_of_1_to_2 and distance_of_3_to_0 are: (" << distance_of_1_to_2 << ", " << distance_of_3_to_0 << ")" << std::endl;
    //std::cout << " ------- debug, ratio1 and ratio2 are: (" << ratio1 << ", " << ratio2 << ")" << std::endl;
    if ((ratio1 >= kRectOpposingSidesMinRatio) && (ratio2 >= kRectOpposingSidesMinRatio)) {
        //两组对边，至少有一组是接近平行状态的(根据透视角度的不同，四边形是一个梯形或者平行四边形)
        //用这个再做一轮判断
        
        int angle_top, angle_bottom, angle_left, angle_right;//4条边和水平轴的夹角
        angle_top = GetAngleOfTwoPoints(rect_points[1], rect_points[0]);
        angle_bottom = GetAngleOfTwoPoints(rect_points[2], rect_points[3]);
        
        angle_right = GetAngleOfTwoPoints(rect_points[2], rect_points[1]);
        angle_left = GetAngleOfTwoPoints(rect_points[3], rect_points[0]);
        
        //std::cout << "\n\n ------- debug, (angle_top, angle_bottom) are: (" << angle_top << ", " << angle_bottom << "), (angle_right, angle_left) are: (" << angle_right << ", " << angle_left << ")" << std::endl;
        
        int diff1 = std::abs(angle_top - angle_bottom);
        int diff2 = std::abs(angle_right - angle_left);
        diff1 = diff1 % 90;
        diff2 = diff2 % 90;//修正到0~90度
        //std::cout << " ---------------debug, diff1 and diff2 are: [" << diff1 << ", " << diff2 << "]" << std::endl;
        
        //这里的几个值，都是经验值
        if (diff1 <= 8 && diff2 <= 8) {
            //俯视拍摄，平行四边形
            return true;
        }
        
        if (diff1 <= 8 && diff2 <= 45) {
            //梯形，有透视效果
            return true;
        }
        if (diff1 <= 45 && diff2 <= 8) {
            //梯形，有透视效果
            return true;
        }
    }
    
    return false;
}


#define ENABLE_DEBUG_MODE
std::tuple<bool, std::vector<cv::Point>, std::vector<cv::Mat> > ProcessEdgeImage(cv::Mat edge_image, cv::Mat color_image, bool draw_debug_image) {
    assert(edge_image.rows == color_image.rows);
    assert(edge_image.cols == color_image.cols);
    
    int height = edge_image.rows;
    int width = edge_image.cols;
    
    std::vector<cv::Point> results;
    std::vector<cv::Mat> debug_images;
    
#ifdef ENABLE_DEBUG_MODE
    cv::Mat lines_image, corners_image, rect_image;
    if (draw_debug_image) {
        lines_image = color_image.clone();
        corners_image = color_image.clone();
        rect_image = color_image.clone();
    }
#endif
    
    /**
     find rectangles
     http://blog.ayoungprogrammer.com/2013/04/tutorial-detecting-multiple-rectangles.html/
     https://github.com/bsdnoobz/opencv-code/blob/master/quad-segmentation.cpp
     http://monkeycoding.com/?p=656
     */
    //<1>0.0~1.0/float类型的image，转换成0~255/int类型
    cv::Mat gray_image;
    edge_image.convertTo(gray_image, CV_8UC1, 255.0);//http://stackoverflow.com/questions/22117267/how-to-convert-an-image-to-a-float-image-in-opencv   http://stackoverflow.com/questions/6302171/convert-uchar-mat-to-float-mat-in-opencv
    
    //<2>找线段
    cv::Mat binary_image;
    threshold(gray_image, binary_image, 128, 255, cv::THRESH_BINARY); //HoughLinesP的输入 mat 是二值化的
    gray_image = binary_image;
    /**
     vector<Vec4i> lines;
     HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
     
     with the arguments:
     
     dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
     lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
     rho : The resolution of the parameter r in pixels. We use 1 pixel.
     theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
     threshold: The minimum number of intersections to “detect” a line
     minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
     maxLineGap: The maximum gap between two points to be considered in the same line.
     */
    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(gray_image, linesP, 1, CV_PI * 1/180, kHoughLinesPThreshold, kHoughLinesPMinLinLength, kHoughLinesPMaxLineGap);//这组参数，可以检测到比较小的矩形，但是对干扰物就更敏感了，很容易检测出矩形框之外的短线段
    
    //<3>线段转换成 参考直线(其实是正好被 image 完整尺寸截断的线段)，并且做一轮过滤
    std::vector<std::tuple<RefLineVec4i, cv::Vec4i> > ref_line_and_segment_pairs;
    for (int i = 0; i < linesP.size(); i++) {
        cv::Vec4i segment = linesP[i];
        RefLineVec4i ref_line = GetRefLine(segment, edge_image.cols, edge_image.rows);//线段延展成 参考线
        
        //线段长度过滤
        double segment_length = cv::sqrt(((float)segment[1] - segment[3]) * ((float)segment[1] - segment[3]) + ((float)segment[0] - segment[2]) * ((float)segment[0] - segment[2]));
        if (segment_length > kHoughLinesPMinLinLength) {
            ref_line_and_segment_pairs.push_back(std::make_tuple(ref_line, segment));
        }
    }
    
    //<4>合并临近的直线
    std::vector<std::tuple<RefLineVec4i, cv::Vec4i> > merged_ref_line_and_segment_pairs = MergeRefLineAndSegmentPairs(ref_line_and_segment_pairs, edge_image.cols, edge_image.rows);
    std::vector<RefLineVec4i> ref_lines;
    std::vector<cv::Vec4i> segments;
    for (int i = 0; i < ref_line_and_segment_pairs.size(); i++) {
        std::tuple<RefLineVec4i, cv::Vec4i> ref_line_and_segment = ref_line_and_segment_pairs[i];
        
        auto ref_line = std::get<0>(ref_line_and_segment);
        auto segment = std::get<1>(ref_line_and_segment);
        
        ref_lines.push_back(ref_line);
        segments.push_back(segment);
    }
 
#ifdef ENABLE_DEBUG_MODE
    if (draw_debug_image) {
        for (int i = 0; i < segments.size(); i++) {
            cv::Vec4i v = segments[i];
            if (draw_debug_image) {
                cv::line(lines_image, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0,255,0));
            }
            //std::cout << " ^^^^^^ debug, cv::Point(v[0], v[1]) is: " << cv::Point(v[0], v[1]) << ",  cv::Point(v[2], v[3]) is: " << cv::Point(v[2], v[3]) << std::endl;
        }
    }
#endif
    
    //<5>寻找segment线段的交叉点以及过滤
    std::vector<cv::Point> all_corners;
    std::vector<Corner> corners;
    for (int i = 0; i < segments.size(); i++) {
        for (int j = i + 1; j < segments.size(); j++) {
            cv::Vec4i segment_a = segments[i], segment_b = segments[j];
            
            //https://gist.github.com/ceykmc/18d3f82aaa174098f145 two lines intersection
            //http://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
            cv::Point intersection_point;
            if (GetSegmentIntersection(segment_a, segment_b, intersection_point) == true) {
                all_corners.push_back(intersection_point);
                
                
                //对交叉点进行第一轮过滤
                if (intersection_point.x <= 0 || intersection_point.y <= 0
                    || intersection_point.x >= width || intersection_point.y >= height) {
                    //std::cout << "^^^^^^^^^^^^^^ pointer <= 0, do not need " << std::endl;
                    //交叉点如果在图片外部，也需要过滤掉
                } else {
                    int thetaA = GetAngleOfLine(segment_a);
                    int thetaB = GetAngleOfLine(segment_b);
                    
                    int angle = std::abs(thetaA - thetaB);
                    angle = angle % 180;//再修正到180度范围内
                    //std::cout << " ------- debug, (thetaA, thetaB) are: (" << thetaA << ", " << thetaB <<  "), two line angle is " << angle << std::endl;
                    
                    if (angle >= kIntersectionMinAngle && angle <= kIntersectionMaxAngle) {
                        //基于两条线的角度进行过滤
                        Corner c = Corner();
                        c.point = intersection_point;
                        c.segments.push_back(segment_a);
                        c.segments.push_back(segment_b);
                        corners.push_back(c);
                    }
                }
            }
        }
    }
    
    //对交叉点进行第二轮过滤，两个点如果很接近，则合并成同一个点，并且用他们的平均值来标示该点
    std::vector<Corner> average_corners;
    for(int i = 0; i < corners.size(); i++) {
        Corner corner = corners[i];
        
        if (average_corners.size() == 0) {
            average_corners.push_back(corner);
        } else {
            bool isCloser = false;
            for (int j = 0; j < average_corners.size(); j++) {
                Corner c = average_corners[j];
                
                cv::Point diff = corner.point - c.point;
                double distance = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
                //std::cout << " _____ debug, distance is: " << distance << std::endl;
                if (distance < kCloserPointMaxDistance) {
                    //两个点很靠近，合并成同一个点
                    Corner newCornet = Corner();
                    newCornet.point = cv::Point((corner.point.x + c.point.x) / 2, (corner.point.y + c.point.y) / 2);
                    
                    //还要合并每个 cornet 的 segment 线段数组
                    std::vector<cv::Vec4i> segment_a = corner.segments;
                    std::vector<cv::Vec4i> segment_b = c.segments;
                    
                    //这种办法合并数组更高效
                    //http://stackoverflow.com/questions/2551775/appending-a-vector-to-a-vector
                    newCornet.segments.insert(newCornet.segments.end(), segment_a.begin(), segment_a.end());
                    newCornet.segments.insert(newCornet.segments.end(), segment_b.begin(), segment_b.end());
                    
                    average_corners[j] = newCornet;
                    isCloser = true;
                    break;
                }
            }
            
            if (isCloser == false) {
                average_corners.push_back(corner);
            }
        }
    }
    //std::cout << "debug, all_corners " << all_corners.size() << ",  corners " << corners.size() << ", average_corners " << average_corners.size() << std::endl;

#ifdef ENABLE_DEBUG_MODE
    if (draw_debug_image) {
        for (int i = 0; i < average_corners.size(); i++) {
            Corner corner = average_corners[i];
            cv::circle(corners_image, corner.point, 3, CV_RGB(255,0,0), 2);
            //std::cout << "average_corners i = " << i << ",  point is: " << corner.point << ", segment size is: " << corner.segments.size() << std::endl;
            
            for (int j = 0; j < corner.segments.size(); j++) {
                cv::Vec4i v = corner.segments[j];
                cv::line(corners_image, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0,0,255));
            }
        }
    }
#endif
    
    //<6>寻找四边形
    if (average_corners.size() >= 4) {
        //至少要有4个点，才算是矩形(TODO，如果点太多，还会影响计算性能，所以可能还需要一个上限值，并且，当达到上限值的时候，还需要考虑如何进一步处理，减少点的数量)
        double maxPerimeter = 0.0;
        
        std::vector<Corner> rect_corners;
        std::vector<Corner> rect_corners_with_max_perimeter;
        std::vector<cv::Point> rect_points_with_max_perimeter;
        
        //4重循环的计算量还是有点大
        for(int i = 0; i <= average_corners.size() - 4; i++) {
            for(int j = i + 1; j <= average_corners.size() - 3; j++) {
                for(int m = j + 1; m <= average_corners.size() - 2; m++) {
                    for(int n = m + 1; n <= average_corners.size() - 1; n++) {
                        
                        rect_corners.clear();
                        rect_corners.push_back(average_corners[i]);
                        rect_corners.push_back(average_corners[j]);
                        rect_corners.push_back(average_corners[m]);
                        rect_corners.push_back(average_corners[n]);
                        
                        //对四个点按照顺时针方向排序
                        rect_corners = ArrangeRectCorners(rect_corners);
                        
                        //如果不是一个合理的四边形，则直接排除
                        if (IsRectCornersReasonable(rect_corners, edge_image.cols) == false) {
                            continue;
                        }
                        
                        std::vector<cv::Point> rect_points;
                        rect_points.push_back(rect_corners[0].point);
                        rect_points.push_back(rect_corners[1].point);
                        rect_points.push_back(rect_corners[2].point);
                        rect_points.push_back(rect_corners[3].point);
                        
                        double perimeter = contourArea(rect_points);//或者用最大面积
                        //double perimeter = arcLength(rect_points, true);//最大周长
                        //std::cout << "#############debug, perimeter is: " << perimeter << std::endl;
                        
                        if (perimeter > maxPerimeter) {
                            maxPerimeter = perimeter;
                            rect_corners_with_max_perimeter = rect_corners;
                            rect_points_with_max_perimeter = rect_points;
                        }
                    }
                }
            }
        }
        
        if (rect_points_with_max_perimeter.size() == 4) {
#ifdef ENABLE_DEBUG_MODE
            if (draw_debug_image) {
                const cv::Point *pts = (const cv::Point*) cv::Mat(rect_points_with_max_perimeter).data;
                int npts = cv::Mat(rect_points_with_max_perimeter).rows;
                
                polylines(rect_image, &pts, &npts, 1,
                          true,
                          cv::Scalar(0, 255, 255),
                          2,
                          CV_AA, 0);
            }
#endif
            
            results = rect_points_with_max_perimeter;
        }
    }
    
#ifdef ENABLE_DEBUG_MODE
    if (draw_debug_image) {
        debug_images.push_back(gray_image);
        debug_images.push_back(lines_image);
        debug_images.push_back(corners_image);
        debug_images.push_back(rect_image);
    }
#endif
    
    bool find_rect = results.size() == 4 ? true : false;
    return std::make_tuple(find_rect, results, debug_images);
}
