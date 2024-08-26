
import os

import cv2
import numpy as np
# from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


def run(args):
    # dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    ids = []

    # ---------------------------------------------------------------------------------------
    root = r'Dataset/RRDSD/SegmentationClassAug'
    with open(r'img_labels_rrdsd/train_aug_building.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root, file[i] + '.png'), cv2.IMREAD_GRAYSCALE)
            if (img == 0).all():
                continue
            ids.append(file[i])
            new_img = img.copy()
            new_img = new_img / 255
            new_img = new_img.astype(int)
            labels.append(new_img)
    # ---------------------------------------------------------------------------------------
    n_images = 0
    for i, id in enumerate(ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']  # high_res指的是high_resolution
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        new_cls_labels = cls_labels.copy()
        preds.append(new_cls_labels)

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)  # 真实值
    resj = confusion.sum(axis=0) # 预测值
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    # fp = 1. - gtj / denominator
    # fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print("threshold:", args.cam_eval_thres, 'iou:', iou, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    # among_predfg_bg表示背景像素被预测为建筑物类的比例，越小越好
    print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))

    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)
    print({'precision': precision, 'recall': recall, 'F_score': F_score})

    return np.nanmean(iou)
