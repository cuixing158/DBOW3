/**
* @file        :demo_general_query.cpp
* @brief       :用于测试是否与MATLAB一致,结论：达到一致
* @details     :与mex/loopDatabase.cpp功能一致，提供"init","load","add","query"四个模式，MATLAB数组或者字符串string可以直接传入本文件使用
* @date        :2023/03/28 10:46:42
* @author      :cuixingxing(cuixingxing150@gmail.com)
* @version     :1.0
*
* @copyright Copyright (c) 2023
*
*/
// OpenCV
#include <iostream>

// DBoW3
#include "loopDatabase_x86_64.h"

int main(int argc, char** argv) {
    try {
        string imageListFile = "/opt_disk2/rd22946/matlab_works/buildMapping/test/imagePathList.txt";  // ls -R /opt_disk2/rd22946/matlab_works/buildMapping/test/database/*.*g>../test/imagePathList.txt
        string descriptor = "orb";
        string databasePath = "./small_db.yml.gz";

        // step_0，产生词袋特征
        loopDatabase_x86_64_init_images(imageListFile.c_str(), databasePath.c_str());

        // step_1,图片特征添加到词袋数据库
        std::string line;
        ifstream fid(imageListFile, std::ios::in);
        while (std::getline(fid, line)) {
            cv::Mat srcImg = cv::imread(line, 0);
            cv::Mat matlabImg = srcImg.t();
            matlabImg = matlabImg.clone();
            loopDatabase_x86_64_add_image(matlabImg.data, srcImg.rows, srcImg.cols);
        }
        fid.close();

        // step_2,检索图
        cv::Mat queryImg = cv::imread("/opt_disk2/rd22946/matlab_works/buildMapping/test/queryImages/query3.jpg", 0);
        std::cout << queryImg.size() << std::endl;
        if (queryImg.empty()) throw std::runtime_error("could not open image");

        cv::Mat matlabIn = queryImg.t();
        matlabIn = matlabIn.clone();
        const unsigned char* inImage = matlabIn.data;
        double queryResult[20] = {0.0};
        loopDatabase_x86_64_query_image(inImage, 480, 640, queryResult);
        std::cout << "result:" << std::endl;
        for (size_t i = 0; i < 10; i++) {
            std::cout << queryResult[i] << "," << queryResult[i + 10] << endl;
        }

    } catch (std::exception& ex) {
        cerr << ex.what() << endl;
    }

    return 0;
}
