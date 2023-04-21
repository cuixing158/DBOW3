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

typedef struct imageViewSt {
    cv::Mat descriptors;
    std::vector<cv::Point2d> keyPoints;
} imageViewSt;
std::vector<cv::Mat> descriptors;
std::vector<std::vector<cv::Point2d> > keypoints;

// 对应OpenCV的cv::Mat转MATLAB uint8类型或logical或者double图像
template <typename T>
void convertCVToMatrix(cv::Mat& srcImg, int rows, int cols, int channels, T* dst) {
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

    // std::cout << "Database information: " << std::endl
    //           << db << std::endl;
    QueryResults ret;
    db.query(descriptors, ret, 10);  // 选取的是top 10

    return ret;
}

QueryResults retrieveFeatures(cv::Mat queryFeatures, Database& db) {
    // std::cout << "Database information: " << std::endl
    //           << db << std::endl;
    QueryResults ret;
    db.query(queryFeatures, ret, 10);  // 选取的是top 10

    return ret;
}

// 剩余函数为供MATLAB使用
void loopDatabase_x86_64_init_images(const char* imageListFile, const char* saveDataBaseYmlGz) {
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

    std::cout << "From images,Create vocabulary,please wait ..." << std::endl;
    DBoW3::Vocabulary voc(k, L, weight, score);
    voc.create(features);
    cout << "... done!" << endl;

    db.setVocabulary(voc, false, 0);  // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    std::string databaseFile(saveDataBaseYmlGz);
    std::cout << "Vocabulary information: " << std::endl
              << voc << std::endl
              << "have saved this path:" << databaseFile << std::endl;
    db.save(databaseFile);
}

void loopDatabase_x86_64_init_features(const unsigned char* inImageOrFeatures, int rows, int cols, bool isOver, const char* saveDataBaseYmlGz) {
    static std::vector<cv::Mat> features;
    if (isOver) {
        // branching factor and depth levels
        const int k = 10;
        const int L = 4;
        const WeightingType weight = TF_IDF;
        const ScoringType score = L1_NORM;

        std::cout << "From features,Create vocabulary,please wait ..." << std::endl;
        DBoW3::Vocabulary voc(k, L, weight, score);
        voc.create(features);

        db.setVocabulary(voc, false, 0);  // false = do not use direct index
        // (so ignore the last param)
        // The direct index is useful if we want to retrieve the features that
        // belong to some vocabulary node.
        // db creates a copy of the vocabulary, we may get rid of "voc" now

        std::string databaseFile(saveDataBaseYmlGz);
        std::cout << "Vocabulary information: " << std::endl
                  << voc << std::endl
                  << "have saved this path:" << databaseFile << std::endl;
        db.save(databaseFile);
        features.clear();
    } else {
        cv::Mat oriFeatures = cv::Mat(cols, rows, CV_8UC1, (void*)inImageOrFeatures);
        assert(cols == 32);
        features.push_back(oriFeatures.t());
    }
}

void loopDatabase_writeStep_imst(const unsigned char* inImageOrFeatures, int rows, int cols, const double* keyptsX, const double* keyptsY, bool isOver, const char* saveImageViewStFile) {
    static std::vector<imageViewSt> imgSt;
    if (isOver) {
        std::cout << "Saving imageViewSt,please wait ..." << std::endl;
        std::string imgStFile(saveImageViewStFile);
        cv::FileStorage fs(imgStFile, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            cerr << "failed to open file：" + imgStFile + " to write." << endl;
        }
        fs << "numbers" << (int)imgSt.size();
        fs << "descriptors"
           << "{";
        for (size_t i = 0; i < imgSt.size(); i++) {
            fs << "descriptors_" + std::to_string(i) << imgSt[i].descriptors;
        }
        fs << "}";
        fs << "keyPoints"
           << "{";
        for (size_t i = 0; i < imgSt.size(); i++) {
            fs << "keyPoints_" + std::to_string(i) << imgSt[i].keyPoints;
        }
        fs << "}";
        std::cout << "Done" << std::endl;
        fs.release();
        imgSt.clear();
    } else {
        cv::Mat oriFeatures = cv::Mat(cols, rows, CV_8UC1, (void*)inImageOrFeatures);
        std::vector<cv::Point2d> keypts;
        for (size_t i = 0; i < rows; i++) {
            keypts.push_back(cv::Point2d(keyptsX[i], keyptsY[i]));
        }
        imgSt.push_back(imageViewSt{oriFeatures.t(), keypts});
    }
}

void loopDatabase_read_imst_numEles(const char* saveImageViewStFile, int* numEles) {
    descriptors.clear();
    keypoints.clear();
    std::string imgStFile(saveImageViewStFile);
    cv::FileStorage fs(imgStFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "failed to open file: " + imgStFile + " to read." << endl;
    }
    fs["numbers"] >> *numEles;
    cv::FileNode descriptorsNode = fs["descriptors"];
    cv::FileNode keyPointsNode = fs["keyPoints"];
    for (auto iter = descriptorsNode.begin(); iter != descriptorsNode.end(); iter++) {
        cv::Mat temp;
        *iter >> temp;
        descriptors.push_back(temp);
    }
    for (auto iter = keyPointsNode.begin(); iter != keyPointsNode.end(); iter++) {
        std::vector<cv::Point2d> temp;
        *iter >> temp;
        keypoints.push_back(temp);
    }
    fs.release();
}

void loopDatabase_readStep_imst_meta(int idx, int* rows, int* cols) {
    idx = idx - 1;
    *rows = descriptors[idx].rows;
    *cols = descriptors[idx].cols;
}

void loopDatabase_readStep_imst(int idx, unsigned char* inImageOrFeatures, double* keyptsXY) {
    idx = idx - 1;
    cv::Mat oriFeat = descriptors[idx];
    convertCVToMatrix(oriFeat, oriFeat.rows, oriFeat.cols, 1, inImageOrFeatures);
    cv::Mat keyPts = cv::Mat(keypoints[idx]).reshape(1);  // NX2 double类型Mat
    convertCVToMatrix(keyPts, keyPts.rows, keyPts.cols, 1, keyptsXY);
}

void loopDatabase_x86_64_load(const char* databaseYmlGz) {
    std::string dbFile(databaseYmlGz);
    std::cout << "loading database,please wait... ,name:" << dbFile << std::endl;
    db.load(dbFile);
    std::cout << "load finished!" << std::endl;
}

// 480*640=307200, 从matlab传入进来的为480*640 单通道uint8图像
void loopDatabase_x86_64_add_image(const unsigned char* inImage, int rows, int cols) {
    // 注意MATLAB数组传入的是以列为优先的，与OpnenCV正好相反
    cv::Mat oriImg = cv::Mat(cols, rows, CV_8UC1, (void*)inImage);

    std::string featureName = "orb";
    cv::Mat feature = loadFeatures(oriImg.t(), featureName);
    db.add(feature);
}

void loopDatabase_x86_64_add_features(const unsigned char* inFeatures, int rows, int cols) {
    cv::Mat matlabORBFeatures = cv::Mat(cols, rows, CV_8UC1, (unsigned char*)inFeatures);
    db.add(matlabORBFeatures.t());
}

// 返回top10，10*2大小数组给MATLAB，第一列为queryID,第二列为score
void loopDatabase_x86_64_query_image(const unsigned char* inImage, int rows, int cols, double queryResult[20]) {
    cv::Mat oriImg = cv::Mat(cols, rows, CV_8UC1, (void*)inImage);

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

// 返回top10，10*2大小数组给MATLAB，第一列为queryID,第二列为score
void loopDatabase_x86_64_query_features(const unsigned char* inFeatures, int rows, int cols, double queryResult[20]) {
    cv::Mat oriFeatures = cv::Mat(cols, rows, CV_8UC1, (void*)inFeatures);

    QueryResults result = retrieveFeatures(oriFeatures.t(), db);

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