/**
* @file        :loopDatabase_x86_64.h
* @brief       :用于在X86-64平台上为生成的C/C++代码直接使用DBOW3代码
* @details     :与mex/loopDatabase.cpp功能一致，提供"init","load","add","query"四个模式，MATLAB数组或者字符串string可以直接传入本文件使用
* @date        :2023/02/16 15:46:42
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/

#ifndef _LOOPDATABASE_X64_
#define _LOOPDATABASE_X64_

// base
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;

// 用于创建database
extern Database db;

/** Function Declarations */
extern void loopDatabase_x86_64_init(const char* imageListFile);

extern void loopDatabase_x86_64_load(const char* databaseYmlGz);

extern void loopDatabase_x86_64_add(const unsigned char inImage[307200]);  // 480*640=307200, 从matlab传入进来的为480*640 单通道uint8图像

extern void loopDatabase_x86_64_query(const unsigned char inImage[307200], double queryResult[20]);  // 返回top10，10*2大小数组给MATLAB，第一列为queryID,第二列为score

#endif