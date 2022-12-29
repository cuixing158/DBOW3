#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        :imagePaths.py
@description :生成从小到大的顺序图片
@date        :2022/12/28 15:57:05
@author      :cuixingxing
@email       :cuixingxing150@gmail.com
@version     :1.0
'''


import os
import functools

# 自定义排序规则


def my_compare(path1, path2):
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)
    if int(name1[4:-4]) < int(name2[4:-4]):
        return -1
    else:
        return 1


imagesFileName = []
# "/opt_disk2/rd22946/my_data/from_tongwenchao/map_R_resize"
imagesDir = "/opt_disk2/rd22946/my_data/bookCovers/database"

for (dirpath, dirnames, filenames) in os.walk(imagesDir):
    for filename in filenames:
        if filename.endswith(".jpg"):
            imagesFileName.append(os.path.join(dirpath, filename))

imagesFileName.sort(key=functools.cmp_to_key(my_compare))
# imagesFileName = sorted(imagesFileName, key=functools.cmp_to_key(my_compare))
with open("imagePath.txt", "w") as fid:
    fid.writelines(name + "\n" for name in imagesFileName)
