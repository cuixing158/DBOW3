/**
* @file        :retrieveImageDBOW3.cpp
* @brief       :用于图像检索，matlab mex使用
* @details     :This is the detail description.
* @date        :2023/01/02 09:50:04
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;

// MATLAB
#include "mex.hpp"
#include "mexAdapter.hpp"

using matlab::mex::ArgumentList;
using namespace matlab::data;

QueryResults retrieveImages(cv::Mat queryImage, std::string dbFile) {
    // load the vocabulary from disk
    //Vocabulary voc(vocFile);

    Database db(dbFile);  // 已经加入了每幅图像的特征的database

    // add images to the database
    cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(2000);
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    if (queryImage.channels() == 3) {
        cv::cvtColor(queryImage, queryImage, cv::COLOR_BGR2GRAY);
    }

    if (queryImage.empty()) throw std::runtime_error("Could not open image");
    fdetector->detectAndCompute(queryImage, cv::Mat(), keypoints, descriptors);

    // cout << "Database information: " << endl
    //      << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    db.query(descriptors, ret, 10);

    return ret;
}
// ----------------------------------------------------------------------------

//

/**
* @brief       This is a brief description
* @details     matlab 中语法为： result = retrieveImageDBOW3(image,dbFile)
* @param[in]   inArgName input argument description.
* @param[out]  outArgName output argument description.
* @return      返回值
* @retval      返回值类型
* @par 标识符
*     保留
* @par 其它
*
* @par 修改日志
*      cuixingxing于2023/01/02创建
*/
class MexFunction : public matlab::mex::Function {
    // Pointer to MATLAB engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // Factory to create MATLAB data arrays
    ArrayFactory factory;

   public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        checkArguments(outputs, inputs);

        matlab::data::TypedArray<uint8_t> queryImg = std::move(inputs[0]);
        std::string dataBaseFile = inputs[1][0];
        std::string descriptor = "orb";
        cv::Size oriImgS = cv::Size(queryImg.getDimensions()[1], queryImg.getDimensions()[0]);

        // step1: convert matlab matrix to opencv Mat
        cv::Mat oriImg;
        bool is3Channels = queryImg.getDimensions()[2] == 3;
        if (is3Channels) {
            oriImg = cv::Mat::zeros(oriImgS, CV_8UC3);
            for (size_t i = 0; i < queryImg.getDimensions()[0]; i++) {
                cv::Vec3b* data = oriImg.ptr<cv::Vec3b>(i);
                for (size_t j = 0; j < queryImg.getDimensions()[1]; j++) {
                    data[j] = cv::Vec3b((uchar)queryImg[i][j][2], (uchar)queryImg[i][j][1], (uchar)queryImg[i][j][0]);
                }
            }
        } else {
            oriImg = cv::Mat::zeros(oriImgS, CV_8UC1);
            for (size_t i = 0; i < queryImg.getDimensions()[0]; i++) {
                uchar* data = oriImg.ptr<uchar>(i);
                for (size_t j = 0; j < queryImg.getDimensions()[1]; j++) {
                    data[j] = (uchar)queryImg[i][j];
                }
            }
        }

        // step2 ,algorithm
        QueryResults result = retrieveImages(oriImg, dataBaseFile);

        //step3, convert to matlab
        TypedArray<double_t> matlabResults = factory.createArray<double>({result.size(), 2});
        QueryResults::const_iterator qit;
        size_t rowIdx = 0;
        for (qit = result.begin(); qit != result.end(); ++qit) {
            matlabResults[rowIdx][0] = (double)qit->Id;
            matlabResults[rowIdx][1] = (double)qit->Score;
            rowIdx++;
        }
        outputs[0] = std::move(matlabResults);
    }

    void displayOnMATLAB(const std::ostringstream& stream) {
        matlabPtr->feval(u"fprintf", 0,
                         std::vector<Array>({factory.createScalar(stream.str())}));
    }

    void displayOnMATLAB(const std::string& str) {
        matlabPtr->feval(u"fprintf", 0,
                         std::vector<Array>({factory.createScalar(str)}));
    }

    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() != 2) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("应输入2个参数")}));
        }

        if (inputs[0].getType() != matlab::data::ArrayType::UINT8) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("The first input is uint8 image")}));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("database 输入路径！")}));
        }

        if (outputs.size() != 1) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({factory.createScalar("Output argument must only one")}));
        }
    }
};
