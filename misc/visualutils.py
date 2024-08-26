import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np


def visual_attention_map(att_maps, name, fn='building'):
    for index, map in enumerate(att_maps):
        if fn == 'building':
            map_nor = F.relu(map) / torch.max(F.relu(map))

            map_nor = F.interpolate(map_nor, (64, 64), mode='bilinear', align_corners=False)
            map_nor = F.interpolate(map_nor, (256, 256), mode='bilinear', align_corners=False)[0, 0, :, :]
            map_nor = map_nor.numpy()

            fg_img = np.uint8(255 * map_nor)
            vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
            cv2.imwrite(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/figure/' + name + fn + str(index) + '.png',
                        vis_result)
        else:
            map_nor = F.relu(map) / torch.max(F.relu(map))
            # print(map_nor.squeeze(0).shape)
            map_nor = F.interpolate(map_nor.squeeze(0), (64, 64), mode='bilinear', align_corners=False)
            map_nor = F.interpolate(map_nor, (256, 256), mode='bilinear', align_corners=False)[0, 0, :, :]
            map_nor = map_nor.cpu().numpy()

            fg_img = np.uint8(255 * map_nor)
            vis_result = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
            cv2.imwrite(r'/mnt/sdb1/zhengdaoyuan/PycharmProjects/AMN/figure/' + name + fn + str(index) + '.png',
                        vis_result)
