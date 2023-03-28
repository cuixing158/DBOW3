/**
* @file        :mainCreateDatabase.cpp
* @brief       :用于产生字典的本地文件，mex编写，可用于matlab中使用。改编自demo_general.cpp
* @details     :This is the detail description.
* @date        :2022/12/30 16:20:31
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2022
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

// MATLAB
#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace DBoW3;
using namespace std;

using matlab::mex::ArgumentList;
using namespace matlab::data;

vector<cv::Mat> loadFeatures(std::vector<string> path_to_images, string descriptor = "") throw(std::exception) {
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
        throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat> features;

    cout << "Extracting   features..." << endl;
    for (size_t i = 0; i < path_to_images.size(); ++i) {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty()) throw std::runtime_error("Could not open image" + path_to_images[i]);
        cout << "extracting features" << endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
    }
    return features;
}

void saveDatabase(const vector<cv::Mat>& features, std::string dataBaseFile) {
    // branching factor and depth levels
    const int k = 10;
    const int L = 4;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl
         << endl;

    Database db(voc, false, 0);  // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for (size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "add features done!" << endl;

    cout << "Database information: " << endl
         << db << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save(dataBaseFile);
    cout << "... done!" << endl;
}
// ----------------------------------------------------------------------------

//

/**
* @brief       用于matlab中创建database
* @details     matlab中语法设计为
*              createDatabase(imageFileList,outputDir)
* @param[in]   imageFileList 存储图像jpg的列表文件，每行存储一副图像完整路径。
* @param[out]  outputDir 指定输出的字典路径，该路径必须存在
* @return      返回值
* @retval      返回值类型
* @par 标识符
*     保留
* @par 其它
*
* @par 修改日志
*      cuixingxing于2022/12/30创建
*/
class MexFunction : public matlab::mex::Function {
    // Pointer to MATLAB engine
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

    // Factory to create MATLAB data arrays
    ArrayFactory factory;

   public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
        checkArguments(outputs, inputs);

        std::string imageListPath = inputs[0][0];
        std::string dataBaseDir = inputs[1][0];

        std::string dataBaseFile = dataBaseDir + "/small_db.yml.gz";
        std::string descriptor = "orb";
        std::vector<string> images;
        string line;
        ifstream fid(imageListPath, std::ios_base::in);
        while (std::getline(fid, line)) {
            images.push_back(line);
        }

        vector<cv::Mat> features = loadFeatures(images, descriptor);
        //testVocCreation(features);

        //testDatabase(features);
        saveDatabase(features, dataBaseFile);
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

        if (inputs[0].getType() != matlab::data::ArrayType::MATLAB_STRING) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("The first input is 文件绝对路径")}));
        }

        if (inputs[1].getType() != matlab::data::ArrayType::MATLAB_STRING) {
            matlabPtr->feval(u"error",
                             0, std::vector<matlab::data::Array>({factory.createScalar("database 输出路径！")}));
        }

        if (outputs.size() != 0) {
            matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({factory.createScalar("Output argument must only one")}));
        }
    }
};
