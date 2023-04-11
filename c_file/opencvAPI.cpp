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

template void convertCVToMatrix<unsigned char>(cv::Mat &, int, int, int, unsigned char *);
template void convertCVToMatrix<bool>(cv::Mat &, int, int, int, bool *);
template void convertCVToMatrix<double>(cv::Mat &, int, int, int, double *);
template void imwarp2<bool>(bool const *, int, int, int, double *, imref2d_ *, bool *);
template void imwarp2<unsigned char>(unsigned char const *, int, int, int, double *, imref2d_ *, unsigned char *);
template void alphaBlendOpenCV<bool, unsigned char>(const unsigned char *, int, int, int, unsigned char const *, bool const *, int, int, int, int, int, unsigned char *);
template void alphaBlendOpenCV<unsigned char, unsigned char>(const unsigned char *, int, int, int, unsigned char const *, unsigned char const *, int, int, int, int, int, unsigned char *);
template void alphaBlendOpenCV<const unsigned char, unsigned char>(const unsigned char *, int, int, int, const unsigned char *, const unsigned char *, int, int, int, int, int, unsigned char *);

// 对应OpenCV的cv::Mat转MATLAB uint8类型或logical或者double图像
template <typename T>
void convertCVToMatrix(cv::Mat &srcImg, int rows, int cols, int channels, T *dst) {
    size_t elems = rows * cols;
    if (channels == 3) {
        cv::Mat channels[3];
        cv::split(srcImg.t(), channels);

        memcpy(dst, channels[2].data, elems * sizeof(T));              //copy channel[2] to the red channel
        memcpy(dst + elems, channels[1].data, elems * sizeof(T));      // green
        memcpy(dst + 2 * elems, channels[0].data, elems * sizeof(T));  // blue
    } else {
        srcImg = srcImg.t();
        memcpy(dst, srcImg.data, elems * sizeof(T));
    }
}

// 对应MATLAB uint8类型或者logical图像转cv::Mat，图像在内存中连续
template <typename T>
void convertToMatContinues(const T *inImg, int rows, int cols, int channels, cv::Mat &matBigImg) {
    size_t elems = (size_t)rows * cols;
    T *array = (T *)inImg;
    if (channels == 3) {
        cv::Mat matR = cv::Mat(cols, rows, CV_8UC1, array);  //inImg在内存中必须连续
        cv::Mat matG = cv::Mat(cols, rows, CV_8UC1, array + elems);
        cv::Mat matB = cv::Mat(cols, rows, CV_8UC1, array + 2 * elems);
        std::vector<cv::Mat> matBGR = {matB.t(), matG.t(), matR.t()};
        cv::merge(matBGR, matBigImg);
    } else {
        matBigImg = cv::Mat(cols, rows, CV_8UC1, (T *)inImg);
        matBigImg = matBigImg.t();
    }
}

// 对应MATLAB uint8类型或者logical图像转cv::Mat，图像在内存中不连续
void convertToMat(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg) {
    int elems = rows * cols;
    if (channels == 3) {
        matBigImg = cv::Mat(rows, cols, CV_8UC3, cv::Scalar::all(0));

        for (int i = 0; i < rows; i++) {
            cv::Vec3b *data = matBigImg.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; j++) {
                data[j][2] = (uchar)inImg[i + rows * j];
                data[j][1] = (uchar)inImg[i + rows * j + elems];
                data[j][0] = (uchar)inImg[i + rows * j + 2 * elems];
            }
        }
    } else {
        matBigImg = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));

        for (int i = 0; i < rows; i++) {
            uchar *data = matBigImg.ptr<uchar>(i);
            for (int j = 0; j < cols; j++) {
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
// const unsigned char/bool ----> logical, coder.rref
// const char ----> string,character vector,coder.rref
// int ---->int32
// double ----> double
template <typename T>
void imwarp2(const T *inImg, int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, T *outImg) {
    cv::Mat srcImg, dstImg;

    convertToMatContinues(inImg, rows, cols, channels, srcImg);
    // may be projective ,https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html#bvnhvs8
    double E = tformA[2];
    double F = tformA[5];
    double matlabOutputViewOffset = 0.5f;
    cv::Mat transMat;
    if (std::abs(E) > 10 * std::numeric_limits<double>::epsilon() || std::abs(F) > 10 * std::numeric_limits<double>::epsilon())  // projective
    {
        transMat = (cv::Mat_<double>(3, 3) << tformA[0], tformA[3], tformA[6],
                    tformA[1], tformA[4], tformA[7],
                    tformA[2], tformA[5], tformA[8]);
        // 平移到可视化区域
        transMat.colRange(2, 3) = transMat.colRange(2, 3) - (cv::Mat_<double>(3, 1) << outputView->XWorldLimits[0] - matlabOutputViewOffset, outputView->YWorldLimits[0] - matlabOutputViewOffset, 0.0);

        cv::warpPerspective(srcImg, dstImg, transMat, cv::Size(outputView->ImageSize[1], outputView->ImageSize[0]));
    } else {
        transMat = (cv::Mat_<double>(2, 3) << tformA[0], tformA[3], tformA[6],
                    tformA[1], tformA[4], tformA[7]);
        // 平移到可视化区域
        transMat.colRange(2, 3) = transMat.colRange(2, 3) - (cv::Mat_<double>(2, 1) << outputView->XWorldLimits[0] - matlabOutputViewOffset, outputView->YWorldLimits[0] - matlabOutputViewOffset);
        cv::warpAffine(srcImg, dstImg, transMat, cv::Size(outputView->ImageSize[1], outputView->ImageSize[0]));
    }

#ifdef USE_OUTPUT_IMAGEREF
    if (transMat.rows == 2) {
        cv::Mat homoMat = (cv::Mat_<double>(1, 3) << 0.0, 0.0, 1.0);
        transMat.push_back(homoMat);
    }
    cv::Mat edgePts = (cv::Mat_<double>(3, 4) << 0.0, static_cast<double>(cols), static_cast<double>(cols), 0.0,
                       0.0, 0.0, static_cast<double>(rows), static_cast<double>(rows),
                       1.0, 1.0, 1.0, 1.0);
    edgePts.rowRange(0, 2) += matlabOutputViewOffset;
    cv::Mat outPts = transMat * edgePts;
    cv::Mat dstPts, temp;
    cv::repeat(outPts.row(2), 3, 1, temp);
    cv::divide(outPts, temp, dstPts);

    cv::minMaxLoc(dstPts.row(0), &ot->XWorldLimits[0], &ot->XWorldLimits[1], NULL, NULL);
    cv::minMaxLoc(dstPts.row(1), &ot->YWorldLimits[0], &ot->YWorldLimits[1], NULL, NULL);
    ot->ImageSize[0] = static_cast<double>(dstImg.rows);
    ot->ImageSize[1] = static_cast<double>(dstImg.cols);
    std::cout << "图像宽宽:" << ot->ImageSize[1] << "图像高度：" << ot->ImageSize[0] << std::endl;
#endif

    convertCVToMatrix(dstImg, dstImg.rows, dstImg.cols, dstImg.channels(), outImg);
}

void imreadOpenCV(const char *imagePath, unsigned char outImg[]) {
    std::string imgPath(imagePath);
    cv::Mat srcImg = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (srcImg.empty()) {
        std::runtime_error("read image:" + imgPath + " is empty!");
    }

    cv::Mat gray;
    cv::cvtColor(srcImg, gray, cv::COLOR_BGR2GRAY);
    convertCVToMatrix(gray, gray.rows, gray.cols, gray.channels(), outImg);
}

template <typename T1, typename T2>
void alphaBlendOpenCV(const unsigned char downImg[], int rows, int cols, int channels, const unsigned char topImg[], const T1 *maskImg, int maskImgRows, int maskImgCols, int maskImgChannels, int startX, int startY, T2 *outImg) {
    cv::Mat matDownImg, matTopImg, matMaskImg;
    convertToMatContinues(downImg, rows, cols, channels, matDownImg);
    convertToMatContinues(topImg, maskImgRows, maskImgCols, maskImgChannels, matTopImg);
    convertToMatContinues(maskImg, maskImgRows, maskImgCols, maskImgChannels, matMaskImg);
    matMaskImg.convertTo(matMaskImg, matMaskImg.type(), 255);                                                // note: matlab logical,{0,1}
    cv::Rect roi = cv::Rect(startX - 1, startY - 1, maskImgCols, maskImgRows) & cv::Rect(0, 0, cols, rows);  // matlab 0-based,c/c++ 1-based
    cv::Rect roiMask = cv::Rect(0, 0, roi.width, roi.height);
    matTopImg(roiMask).copyTo(matDownImg(roi), matMaskImg(roiMask));
    convertCVToMatrix(matDownImg, rows, cols, channels, outImg);
}
