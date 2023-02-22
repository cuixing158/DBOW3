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
#include "opencv2/opencv.hpp"

typedef struct imref2d {
    double XWorldLimits[2];
    double YWorldLimits[2];
    double ImageSize[2];
} imref2d_;

void imwarp(const cv::Mat srcImg, int rows, int cols, int channels, float tformA[9], imref2d outputView, cv::Mat &outImg);

void imwarp2(const unsigned char inImg[], int rows, int cols, int channels, double tformA[9], imref2d_ *outputView, unsigned char outImg[]);
#endif