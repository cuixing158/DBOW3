# DBOW3回环检测库(图像相似度检测)

# Overview

用于SLAM中回环检测，返回图片的相似度top-k排序，本库支持MATLAB中`mex`文件构建，C/C++代码生成直接映射到此库源代码，使用方便。

## 说明

本库程序绝大部分来源于[官方](./README_official.md)的repo，额外增加了`mex/`、`c_file/`文件目录，分别用于`mex`构建和C/C++代码生成的入口函数。
