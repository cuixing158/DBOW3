/**
* @file        :opencvAPI.CPP
* @brief       :用于在各个嵌入式平台/x86-64上为生成的C/C++代码直接使用OpenCV代码
* @details     :This is the detail description.
* @date        :2023/02/20 16:23:24
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/

#include "opencvAPI.h"

// "marshalling"
void convertCVToMatrix(cv::Mat &srcImg, int rows, int cols, int channels, unsigned char dst[]);

//"marshalling"
void convertToMat(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg);

void convertToMatContinues(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg);

// 对应OpenCV的cv::Mat转MATLAB uint8类型或logical图像
void convertCVToMatrix(cv::Mat &srcImg, int rows, int cols, int channels, unsigned char dst[]) {
    CV_Assert(srcImg.type() == CV_8UC1 || srcImg.type() == CV_8UC3);
    size_t elems = rows * cols;
    if (channels == 3) {
        cv::Mat channels[3];
        cv::split(srcImg.t(), channels);

        memcpy(dst, channels[2].data, elems * sizeof(unsigned char));              //copy channel[2] to the red channel
        memcpy(dst + elems, channels[1].data, elems * sizeof(unsigned char));      // green
        memcpy(dst + 2 * elems, channels[0].data, elems * sizeof(unsigned char));  // blue
    } else {
        srcImg = srcImg.t();
        memcpy(dst, srcImg.data, elems * sizeof(unsigned char));
    }
}

// 对应MATLAB uint8类型或者logical图像转cv::Mat，图像在内存中连续
void convertToMatContinues(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg) {
    size_t elems = (size_t)rows * cols;
    // unsigned char *array = &inImg[0];
    unsigned char *array = (unsigned char *)inImg[0];
    if (channels == 3) {
        cv::Mat matR = cv::Mat(cols, rows, CV_8UC1, array);  //inImg在内存中必须连续
        cv::Mat matG = cv::Mat(cols, rows, CV_8UC1, array + elems);
        cv::Mat matB = cv::Mat(cols, rows, CV_8UC1, array + 2 * elems);
        std::vector<cv::Mat> matBGR = {matB.t(), matG.t(), matR.t()};
        cv::merge(matBGR, matBigImg);
    } else {
        matBigImg = cv::Mat(cols, rows, CV_8UC1, inImg[0]);
        matBigImg = matBigImg.t();
    }
}

// 对应MATLAB uint8类型或者logical图像转cv::Mat，图像在内存中不连续
void convertToMat(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg) {
    size_t elems = (size_t)rows * cols;
    if (channels == 3) {
        matBigImg = cv::Mat(rows, cols, CV_8UC3, cv::Scalar::all(0));

        for (size_t i = 0; i < rows; i++) {
            cv::Vec3b *data = matBigImg.ptr<cv::Vec3b>(i);
            for (size_t j = 0; j < cols; j++) {
                data[j][2] = (uchar)inImg[i + rows * j];
                data[j][1] = (uchar)inImg[i + rows * j + elems];
                data[j][0] = (uchar)inImg[i + rows * j + 2 * elems];
            }
        }
    } else {
        matBigImg = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));

        for (size_t i = 0; i < rows; i++) {
            uchar *data = matBigImg.ptr<uchar>(i);
            for (size_t j = 0; j < cols; j++) {
                data[j] = (uchar)inImg[i + rows * j];
            }
        }
    }
}
/**
* @brief       针对rigidtform2d，affinetform2d等价的imwarp函数
* @details     This is the detail description.
* @param[in]   inArgName input argument description.
* @param[out]  outArgName output argument description.
* @return      返回值
* @retval      返回值类型
* @par 标识符
*     保留
* @par 其它
*
* @par 修改日志
*      cuixingxing于2022/11/12创建
*/
void imwarp(const cv::Mat srcImg, int rows, int cols, int channels, double tformA[9], imref2d_ outputView, cv::Mat &outImg) {
    // CV_Assert(sizeof(inImg) / sizeof(inImg[0]) == rows * cols * channels);  //图像大小要对应
    // CV_Assert(sizeof(tformA) / sizeof(tformA[0]) == 9);                     // 转换矩阵为3*3大小，列优先"
    cv::Mat transMat = (cv::Mat_<double>(2, 3) << tformA[0], tformA[3], tformA[6],
                        tformA[1], tformA[4], tformA[7]);

    // 计算包含目标图像的最大范围

    std::vector<cv::Point2f> srcCorners = {cv::Point2f(0, 0), cv::Point2f(srcImg.cols, 0), cv::Point2f(srcImg.cols, srcImg.rows), cv::Point2f(0, srcImg.rows)};
    std::vector<cv::Point2f> dstCorners;
    cv::transform(srcCorners, dstCorners, transMat);  // 对应matlab的transpointsforward
    dstCorners.insert(dstCorners.end(), srcCorners.begin(), srcCorners.end());
    cv::Rect outputViewRect = cv::boundingRect(dstCorners);
    // 平移到可视化区域
    transMat.colRange(2, 3) = transMat.colRange(2, 3) - (cv::Mat_<double>(2, 1) << outputView.XWorldLimits[0], outputView.YWorldLimits[0]);

    cv::warpAffine(srcImg, outImg, transMat, cv::Size(outputView.ImageSize[1], outputView.ImageSize[0]));
}

// C类型与MATLAB内置类型对应
// const unsigned char ---->uint8, coder.rref
// const unsigned char ----> logical, coder.rref
// const char ----> string,character vector,coder.rref
// int ---->int32
// double ----> double
void imwarp2(const unsigned char inImg[], int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, unsigned char outImg[]) {
    cv::Mat srcImg, dstImg;

    convertToMat(inImg, rows, cols, channels, srcImg);
    // may be projective ,https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html#bvnhvs8
    double E = tformA[2];
    double F = tformA[5];
    if (std::abs(E) > 10 * std::numeric_limits<double>::epsilon() || std::abs(F) > 10 * std::numeric_limits<double>::epsilon())  // projective
    {
        cv::Mat transMat = (cv::Mat_<double>(3, 3) << tformA[0], tformA[3], tformA[6],
                            tformA[1], tformA[4], tformA[7],
                            tformA[2], tformA[5], tformA[8]);
        // 平移到可视化区域
        transMat.colRange(2, 3) = transMat.colRange(2, 3) - (cv::Mat_<double>(3, 1) << outputView->XWorldLimits[0], outputView->YWorldLimits[0], 0.0);

        cv::warpPerspective(srcImg, dstImg, transMat, cv::Size(outputView->ImageSize[1], outputView->ImageSize[0]));
    } else {
        cv::Mat transMat = (cv::Mat_<double>(2, 3) << tformA[0], tformA[3], tformA[6],
                            tformA[1], tformA[4], tformA[7]);
        // 平移到可视化区域
        transMat.colRange(2, 3) = transMat.colRange(2, 3) - (cv::Mat_<double>(2, 1) << outputView->XWorldLimits[0], outputView->YWorldLimits[0]);
        cv::warpAffine(srcImg, dstImg, transMat, cv::Size(outputView->ImageSize[1], outputView->ImageSize[0]));
    }
    convertCVToMatrix(dstImg, dstImg.rows, dstImg.cols, dstImg.channels(), outImg);
}
