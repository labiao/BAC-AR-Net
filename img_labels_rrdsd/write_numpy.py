#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : write_numpy.py
# Author: 郑道远
# Date  : 2021/4/28
import os

import cv2
import numpy as np


def count_building(label, txt_path):
    # f = open(txt_path, "w+")
    # 1.读取标签路径下的掩膜文件
    for _, _, filenames in os.walk(label):
        # 2.循环遍历并统计每个png文件中属于mask = 1的数量
        count1 = count2 = 0
        e = dict()
        for i in range(0, len(filenames)):
            png = os.path.join(label, filenames[i])  # png
            img = cv2.imread(png, -1)

            if img.shape[0] == 256 and img.shape[1] == 256:
                print(img.shape)
                No_building = np.sum(img == 0)  # 统计背景像素
                building = np.sum(img != 0)  # 统计建筑物像素
                # 如果当前的 building > Nobuilding，则将该图放置1文件夹中
                if building >= 256 * 256 / 4:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 1\n")
                    e[filenames[i].split('.')[0]] = [1,]
                    count1 += 1
                # 否则，则将该图放置0文件夹中
                elif building == 0:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 0\n")
                    e[filenames[i].split('.')[0]] = [0,]
                    count2 += 1
            else:
                assert img.shape[0] == 250 and img.shape[1] == 250
                print(img.shape)
                No_building = np.sum(img == 0)  # 统计背景像素
                building = np.sum(img != 0)  # 统计建筑物像素
                # 如果当前的 building > Nobuilding，则将该图放置1文件夹中
                if building >= 250 * 250 / 4:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 1\n")
                    e[filenames[i].split('.')[0]] = [1, ]
                    count1 += 1
                # 否则，则将该图放置0文件夹中
                elif building == 0:
                    # shutil.copyfile(png, os.path.join(save_label, filenames[i]))
                    # f.writelines(filenames[i].split(".")[0] + " 0\n")
                    e[filenames[i].split('.')[0]] = [0, ]   
                    count2 += 1

        # f.close()
        np.save(txt_path, e)
        print(count1, count2)

label = '../Dataset/RRDSD/SegmentationClassAug'
txt_path ='./cls_labels.npy'
count_building(label, txt_path)