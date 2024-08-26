import os

import cv2
import numpy as np
# from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


def visual_refined_unary(refined_unary, name):
    fg_img = np.uint8(255 * refined_unary[0])

    vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
    cv2.imwrite('/mnt/sdb1/zhengdaoyuan/devdata/AMN_results/amn_cam/' + name + '.jpg', vis_result)


def run():
    ids = []

    # ---------------------------------------------------------------------------------------
    root = r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/Dataset/vai/SegmentationClassAug'
    with open(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/img_labels/train_aug_building.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root, file[i] + '.png'), cv2.IMREAD_GRAYSCALE)
            if (img == 0).all():
                continue
            ids.append(file[i])
    # ---------------------------------------------------------------------------------------
    n_images = 0
    for i, id in enumerate(ids):
        n_images += 1
        # cam_dict = np.load(os.path.join(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/result/lpcam_mask', id + '.npy'),
        #                    allow_pickle=True).item()
        # cams = cam_dict['high_res']  # high_res指的是high_resolution
        # if id == 'top_mosaic_09cm_area32_4_9':
        cam_dict = np.load(os.path.join(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/result/amn_cam', id + '.npy'),
                           allow_pickle=True).item()
        # print(cam_dict['high_res'].shape) # (2, 256, 256)
        cams = cam_dict['high_res'][1:, ...]
        # cams = cam_dict['cam'][1:, ...].numpy()

        visual_refined_unary(cams, id)


run()
