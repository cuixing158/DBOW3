

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

QueryResults retrieveImages(cv::Mat queryImage, std::string dbFile) {
    // load the vocabulary from disk
    //Vocabulary voc(vocFile);

    Database db(dbFile);  // 已经加入了每幅图像的特征的database

    // add images to the database
    cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
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
    db.query(descriptors, ret, 4);

    return ret;
}

int main(int argc, char **argv) {
    try {
        string projectDir = "/opt_disk2/rd22946/vscode_work/cppProjects/fbow-master/";
        string descriptor = "orb";
        string outputFile = projectDir + "data/myorbImgsFeatures.feat";
        string databasePath = projectDir + "data/small_db.yml.gz";

        cv::Mat queryImg = cv::imread("/opt_disk2/rd22946/my_data/bookCovers/queries/query3.jpg", 0);
        if (queryImg.empty()) throw std::runtime_error("could not open image");
        QueryResults ret = retrieveImages(queryImg, databasePath);
        std::cout << ret << endl;
    } catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

    return 0;
}
