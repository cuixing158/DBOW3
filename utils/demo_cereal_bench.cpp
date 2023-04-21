/**
* @file        :demo_cereal_bench.cpp
* @brief       :用于测试cereal序列化库的性能，保存文件大小和时间。结论：序列化的数据大小为110M，耗时约0.5秒，具体参考主分支进展。
* @details     :This is the detail description.
* @date        :2023/04/21 10:27:33
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/
#include "opencv2/opencv.hpp"
#include <iostream>

// cereal header files
#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/vector.hpp"
#include <fstream>

// 针对cv::Mat, cv::Point2d类型定义序列化,https://blog.csdn.net/ShuerSu/article/details/121597289
typedef struct imgInfo {
    int imgWidth;
    int imgHeight;
    int imgChannel;
    int imgType;
    std::vector<uint8_t> imgData;
    //结构体中必须包含serialize函数才能使用cereal进行序列化
    template <class Archive>
    void serialize(Archive& ar) {
        ar(imgWidth, imgHeight, imgChannel, imgType, imgData);
    }
} imgInfo;

typedef struct point {
    double x;
    double y;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(x, y);
    }
} point;

template <class Archive>
void serialize(Archive& ar, cv::Point2d& pt) {
    ar(pt.x, pt.y);
}

typedef struct imageViewSt {
    imgInfo descriptors;
    std::vector<point> keyPoints;

    template <class Archive>
    void serialize(Archive& ar) {
        ar(descriptors, keyPoints);
    }
} imageViewSt;

//https://blog.csdn.net/guyuealian/article/details/80253066
template <typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat& mat) {
    return (std::vector<_Tp>)(mat.reshape(1, 1));  //通道数不变，按行转为一行
}
/****************** vector转Mat *********************/
template <typename _Tp>
cv::Mat convertVector2Mat(std::vector<_Tp> v, int channels, int rows) {
    cv::Mat mat = cv::Mat(v);                            //将vector变成单列的mat
    cv::Mat dest = mat.reshape(channels, rows).clone();  //PS：必须clone()一份，否则返回出错
    return dest;
}

int main() {
    {
        std::ofstream os("out.cereal", std::ios::binary);  // 打开标准输出流
        cereal::PortableBinaryOutputArchive archive(os);   // 构建cereal对象，并用os初始化

        cv::RNG rng;
        imageViewSt myData[1198];
        for (size_t i = 0; i < 1198; i++) {
            cv::Mat features = cv::Mat::zeros(2000, 32, CV_8UC1);
            cv::Mat ptsMat = cv::Mat::zeros(2000, 2, CV_64FC1);
            rng.fill(features, cv::RNG::UNIFORM, 0, 255);

            imgInfo outInfo;
            outInfo.imgWidth = features.cols;
            outInfo.imgHeight = features.rows;
            outInfo.imgChannel = features.channels();
            outInfo.imgType = features.type();
            outInfo.imgData.assign(features.data, features.data + features.total() * features.channels());

            myData[i].descriptors = outInfo;
            for (size_t j = 0; j < ptsMat.rows; j++) {
                point pt;
                pt.x = ptsMat.at<double>(j, 0);
                pt.y = ptsMat.at<double>(j, 1);
                myData[i].keyPoints.push_back(pt);
            }

            // myData[i].keyPoints.assign(ptsMat.data, ptsMat.data + ptsMat.rows * ptsMat.cols * ptsMat.channels());
            // myData[i].keyPoints = convertMat2Vector<point>(ptsMat);//error: ‘type’ is not a member of ‘cv::DataType<point>’
        }
        double t1 = cv::getTickCount();
        archive(myData);  // 对标准库的任意类型皆可序列化
        double t2 = cv::getTickCount();

        cv::Mat showData = convertVector2Mat<uint8_t>(myData[0].descriptors.imgData, 1, 2000);
        std::cout << "elapsed seconds:" << (t2 - t1) * 1.0 / cv::getTickFrequency() << ",write data:" << showData.rowRange(0, 5) << std::endl;
    }

    {
        double t1 = cv::getTickCount();
        std::ifstream is("out.cereal", std::ios::binary);
        cereal::PortableBinaryInputArchive iarchive(is);  // Create an input archive

        imageViewSt myDataRead[1198];
        iarchive(myDataRead);  // Read the data from the archive
        double t2 = cv::getTickCount();

        cv::Mat readMat = convertVector2Mat<uint8_t>(myDataRead[0].descriptors.imgData, 1, 2000);
        std::cout << "elapsed seconds:" << (t2 - t1) * 1.0 / cv::getTickFrequency() << ",read data:" << readMat.rowRange(0, 5) << std::endl;
    }
    return 0;
}