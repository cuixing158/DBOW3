/**
* @file        :opencvAPI.h
* @brief       :用于在各个嵌入式平台/x86-64上为生成的C/C++代码直接使用OpenCV代码
* @details     :This is the detail description.
* @date        :2023/02/20 16:23:24
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/

#ifndef _OPENCVAPI_ALLPLATFORM_
#define _OPENCVAPI_ALLPLATFORM_

// base
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

typedef struct imref2d_ {
    double XWorldLimits[2];
    double YWorldLimits[2];
    double ImageSize[2];
} imref2d_;

typedef struct outputImgAndImref2d {
    imref2d_ ref2d;
    unsigned char outImg[];
} outputImgAndImref2d;

// "marshalling"
template <typename T>
void convertCVToMatrix(cv::Mat &srcImg, int rows, int cols, int channels, T *dst);

//"marshalling"
void convertToMat(const unsigned char inImg[], int rows, int cols, int channels, cv::Mat &matBigImg);

template <typename T>
void convertToMatContinues(const T *inImg, int rows, int cols, int channels, cv::Mat &matBigImg);

// void convertToMatContinues(const bool inImg[], int rows, int cols, int channels, cv::Mat &matBigImg);

void imwarp(const cv::Mat srcImg, int rows, int cols, int channels, double tformA[9], imref2d_ outputView, cv::Mat &outImg);

#ifdef USE_OUTPUT_IMAGEREF
// void imwarp2(const unsigned char inImg[], int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, outputImgAndImref2d *ot);
void imwarp2(const unsigned char inImg[], int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, unsigned char outImg[], imref2d_ *ot);
#endif

template <typename T>
void imwarp2(const T *inImg, int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, T *outImg);

void imreadOpenCV(const char *imagePath, unsigned char outImg[]);

template <typename T1, typename T2>
void alphaBlendOpenCV(const unsigned char downImg[], int rows, int cols, int channels, const unsigned char topImg[], const T1 *maskImg, int maskImgRows, int maskImgCols, int maskImgChannels, int startX, int startY, T2 *outImg);

#endif