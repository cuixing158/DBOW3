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

#include "loopDatabase_x86_64.h"

Database db;

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

    std::cout << "Extracting   features..." << std::endl;
    for (size_t i = 0; i < path_to_images.size(); ++i) {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        std::cout << "reading image: " << path_to_images[i] << std::endl;

        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty()) throw std::runtime_error("Could not open image" + path_to_images[i]);
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        std::cout << "done detecting features" << std::endl;
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

    std::cout << "Database information: " << std::endl
              << db << std::endl;
    QueryResults ret;
    db.query(descriptors, ret, 10);  // 选取的是top 10

    return ret;
}

// 剩余函数为供MATLAB使用
void loopDatabase_x86_64_init(const char* imageListFile) {
    std::vector<string> images;
    std::string line;
    ifstream fid(imageListFile, std::ios::in);
    while (std::getline(fid, line)) {
        images.push_back(line);
    }
    fid.close();

    std::string featureName = "orb";
    std::vector<cv::Mat> features = loadFeatures(images, featureName);

    // branching factor and depth levels
    const int k = 10;
    const int L = 4;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    std::cout << "Create vocabulary,please wait ..." << std::endl;
    DBoW3::Vocabulary voc(k, L, weight, score);
    voc.create(features);

    db.setVocabulary(voc, false, 0);  // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    std::cout << "Vocabulary information: " << std::endl
              << voc << std::endl;
    db.save("database.yml.gz");
}

void loopDatabase_x86_64_load(const char* databaseYmlGz) {
    std::string dbFile(databaseYmlGz);
    std::cout << "loading database,please wait... ,name:" << dbFile << std::endl;
    db.load(dbFile);
    std::cout << "load finished!" << std::endl;
}

// 480*640=307200, 从matlab传入进来的为480*640 单通道uint8图像
void loopDatabase_x86_64_add(const unsigned char inImage[307200]) {
    int rows = 640;  // 注意MATLAB数组传入的是以列为优先的，与OpnenCV正好相反
    int cols = 480;
    cv::Mat oriImg = cv::Mat(rows, cols, CV_8UC1, (void*)inImage);

    std::string featureName = "orb";
    cv::Mat feature = loadFeatures(oriImg.t(), featureName);
    db.add(feature);
}

// 返回top10，10*2大小数组给MATLAB，第一列为queryID,第二列为score
void loopDatabase_x86_64_query(const unsigned char inImage[307200], double queryResult[20]) {
    int rows = 640;  // 注意MATLAB数组传入的是以列为优先的，与OpnenCV正好相反
    int cols = 480;
    cv::Mat oriImg = cv::Mat(rows, cols, CV_8UC1, (void*)inImage);

    QueryResults result = retrieveImages(oriImg.t(), db);

    //step3, convert to matlab
    // TypedArray<double_t> matlabResults = factory.createArray<double>({result.size(), 2});
    QueryResults::const_iterator qit;
    size_t rowIdx = 0;
    for (qit = result.begin(); qit != result.end(); ++qit) {
        queryResult[rowIdx] = (double)qit->Id + 1;      // matlab 索引从1开始
        queryResult[rowIdx + 10] = (double)qit->Score;  // 注意前面程序用的是top10
        rowIdx++;
    }
}
