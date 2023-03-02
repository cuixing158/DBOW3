/**
* @file        :loopDatabase.cpp
* @brief       :类似于matlab的bag of features，invertedImageIndex，逐张添加图像特征用于检索
* @details     :详细使用方法见mexFunction说明。本脚本测试通过！
* @date        :2023/01/02 11:48:40
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

//
/**
* @brief       用于loop closure检索
* @details     matlab语法为：
*              loopDatabase(imageFileList,"init"); % 对imageFileList文件中每行图像文件进行特征提取和用于创建词袋，仅执行一次即可
*              loopDatabase(dbFile,"load"); % 对dbFile文件加载为database，仅执行一次即可
*              loopDatabase(image,"add"); % 用于循环中，不断添加图像image的特征
*              result = loopDatabase(image,"query"); % 用于循环中，用于适当时候做检索任务,result为10*2大小矩阵，每行形如[queryID,score]。
*              loopDatabase("aaa","unclock");% 可取消mexUnlock占用，删除mex文件
*      
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

    // 用于创建database
    Database db;

   public:
    MexFunction() {
        std::ostringstream stream;
        stream << "init MexFunction()" << endl;
        displayOnMATLAB(stream);
        mexLock();
    }

    ~MexFunction() {
        std::ostringstream stream;
        stream << "~MexFunction()" << endl;
        displayOnMATLAB(stream);
        db.clear();
    }

    void operator()(ArgumentList outputs, ArgumentList inputs) {
        checkArguments(outputs, inputs);

        std::string flag = inputs[1][0];
        std::string featureName = "orb";
        if (flag == "init") {
            ostringstream ss;

            std::string imageFileList = inputs[0][0];
            std::vector<string> images;
            string line;
            ifstream fid(imageFileList, std::ios_base::in);
            while (std::getline(fid, line)) {
                images.push_back(line);
            }
            fid.close();

            std::vector<cv::Mat> features = loadFeatures(images, featureName);

            // branching factor and depth levels
            const int k = 10;
            const int L = 4;
            const WeightingType weight = TF_IDF;
            const ScoringType score = L1_NORM;

            ss << "Create vocabulary,please wait ..." << endl;
            displayOnMATLAB(ss);
            DBoW3::Vocabulary voc(k, L, weight, score);
            voc.create(features);

            db.setVocabulary(voc, false, 0);  // false = do not use direct index
            // (so ignore the last param)
            // The direct index is useful if we want to retrieve the features that
            // belong to some vocabulary node.
            // db creates a copy of the vocabulary, we may get rid of "voc" now

            ss << "Vocabulary information: " << endl
               << voc << endl;
            displayOnMATLAB(ss);
            db.save("database.yml.gz");
        } else if (flag == "load") {
            std::string dbFile = inputs[0][0];
            db.load(dbFile);
        } else if (flag == "add" || flag == "query") {
            matlab::data::TypedArray<uint8_t> queryImg = std::move(inputs[0]);
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
            if (flag == "add") {
                cv::Mat feature = loadFeatures(oriImg, featureName);
                db.add(feature);
            } else {
                QueryResults result = retrieveImages(oriImg, db);

                //step3, convert to matlab
                TypedArray<double_t> matlabResults = factory.createArray<double>({result.size(), 2});
                QueryResults::const_iterator qit;
                size_t rowIdx = 0;
                for (qit = result.begin(); qit != result.end(); ++qit) {
                    matlabResults[rowIdx][0] = (double)qit->Id + 1;  // matlab 索引从1开始
                    matlabResults[rowIdx][1] = (double)qit->Score;
                    rowIdx++;
                }
                outputs[0] = std::move(matlabResults);
            }
        } else {
            mexUnlock();
        }
    }

    vector<cv::Mat> loadFeatures(std::vector<string> path_to_images, string descriptor = "orb") {
        //select detector
        cv::Ptr<cv::Feature2D> fdetector;
        if (descriptor == "orb")
            fdetector = cv::ORB::create(2000);
        else if (descriptor == "brisk")
            fdetector = cv::BRISK::create(2000);
#ifdef OPENCV_VERSION_3
        else if (descriptor == "akaze")
            fdetector = cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
        else if (descriptor == "surf")
            fdetector = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

        else
            throw std::runtime_error("Invalid descriptor1");
        assert(!descriptor.empty());
        vector<cv::Mat> features;

        ostringstream ss;
        ss << "Extracting   features..." << endl;
        displayOnMATLAB(ss);
        for (size_t i = 0; i < path_to_images.size(); ++i) {
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            ss << "reading image: " << path_to_images[i] << endl;
            displayOnMATLAB(ss);
            cv::Mat image = cv::imread(path_to_images[i], 0);
            if (image.empty()) throw std::runtime_error("Could not open image" + path_to_images[i]);
            fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
            features.push_back(descriptors);
            ss << "done detecting features" << endl;
            displayOnMATLAB(ss);
        }
        return features;
    }

    cv::Mat loadFeatures(cv::Mat srcImg, string descriptor = "orb") {
        //select detector
        cv::Ptr<cv::Feature2D> fdetector;
        if (descriptor == "orb")
            fdetector = cv::ORB::create(2000);
        else if (descriptor == "brisk")
            fdetector = cv::BRISK::create(2000);
#ifdef OPENCV_VERSION_3
        else if (descriptor == "akaze")
            fdetector = cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
        else if (descriptor == "surf")
            fdetector = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

        else
            throw std::runtime_error("Invalid descriptor2");
        assert(!descriptor.empty());

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cv::Mat image = srcImg;
        if (srcImg.channels() == 3)
            cv::cvtColor(srcImg, image, cv::COLOR_BGR2GRAY);

        if (image.empty()) throw std::runtime_error("image is empty!");
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);

        return descriptors;
    }

    QueryResults retrieveImages(cv::Mat queryImage, Database& db) {
        // load the vocabulary from disk
        //Vocabulary voc(vocFile);
        // Database db(dbFile);  // 已经加入了每幅图像的特征的database

        // add images to the database
        cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(2000);
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        if (queryImage.channels() == 3) {
            cv::cvtColor(queryImage, queryImage, cv::COLOR_BGR2GRAY);
        }

        if (queryImage.empty()) throw std::runtime_error("Could not open image");
        fdetector->detectAndCompute(queryImage, cv::Mat(), keypoints, descriptors);

        ostringstream ss;
        ss << "Database information: " << endl
           << db << endl;
        displayOnMATLAB(ss);
        QueryResults ret;
        db.query(descriptors, ret, 10);  // 选取的是top 10

        return ret;
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

        if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("database 输入路径！")}));
        }
        std::string mode = inputs[1][0];
        if (mode == "add" || mode == "query") {
            if (inputs[0].getType() != matlab::data::ArrayType::UINT8) {
                matlabPtr->feval(u"error",
                                 0, std::vector<matlab::data::Array>({factory.createScalar("The first input is uint8 image")}));
            }
        }

        // if (outputs.size() != 1) {
        //     matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({factory.createScalar("Output argument must only one")}));
        // }
    }
};
