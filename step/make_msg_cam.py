import importlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import voc12.dataloader
from misc import torchutils, imutils
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader

cudnn.enabled = True


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    feature_maps = []
    gradients = []

    def save_feature_map(module, input, output):
        feature_maps.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # model.stage1.register_forward_hook(save_feature_map)
    # model.stage1.register_backward_hook(save_gradient)
    # model.stage2.register_forward_hook(save_feature_map)
    # model.stage2.register_backward_hook(save_gradient)
    # model.stage3.register_forward_hook(save_feature_map)
    # model.stage3.register_backward_hook(save_gradient)
    # model.stage4.register_forward_hook(save_feature_map)
    # model.stage4.register_backward_hook(save_gradient)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_up_size = imutils.get_strided_up_size(size, 16)

            n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))
            if n_classes == 0:
                # print('类别为0')
                continue
            oo = []
            for img in pack['img']:
                # img = pack['img'][0]
                outputs = model(img[0].cuda(non_blocking=True))

                #############################gradcam###################################################
                # logits = model(img[0].cuda(non_blocking=True))
                # for i in range(img[0].size(0)):  # 循环 ori 和 ori_flip
                #     model.backward(i)
                #     gradcams = []
                #     # feature_maps 和 gradients 加入列表的顺序刚好倒序
                #     # print(len(feature_maps), len(gradients), gradients[0].shape,
                #     # gradients[1].shape, gradients[2].shape, gradients[3].shape)
                #     for index in range(4):
                #         gradcam = model.generate_cam(i, [feature_maps[index]], [gradients[3-index]])
                #         gradcams.append(gradcam)  # 4个 来自4级特征
                #     outputs.append(gradcams)
                # cams = []
                # for index in range(4):
                #     out = outputs[0][index] + outputs[1][index].flip(-1)
                #     # 归一化处理
                #     cam_min = out.min()
                #     cam_max = out.max()
                #
                #     if cam_max != cam_min:
                #         # print(cam_max, cam_min)
                #         cam = (out - cam_min) / (cam_max - cam_min)
                #         # print(cam.max(), cam.min())
                #         cams.append(cam)
                #     else:
                #         cams.append(torch.zeros_like(out))
                #
                # outputs = cams
                # # print(outputs[0].max(), outputs[1].max(), outputs[2].max())
                # # torch.Size([1, 64, 64]) torch.Size([1, 32, 32]) torch.Size([1, 16, 16]) torch.Size([1, 16, 16])
                # feature_maps = []
                # gradients = []
                #############################gradcam###################################################

                highres_cam = [F.interpolate(torch.unsqueeze(o.detach(), 1), strided_up_size,
                                             mode='bilinear', align_corners=False) for o in outputs]  # 4个级别输出
                highres_cam_norm = []
                for hc in highres_cam:
                    valid_cat = torch.nonzero(label)[:, 0] + 1
                    hc = hc[valid_cat]
                    hc /= F.adaptive_max_pool2d(hc, (1, 1)) + 1e-5
                    highres_cam_norm.append(hc)
                # 确保权重扩展到与张量相同的维度
                # weights = torch.tensor([0, 0, 0, 1]).cuda()
                weights = torch.tensor([1 / 3, 1 / 3, 1 / 3, 1]).cuda()
                weights = weights.view(-1, 1, 1, 1)
                # 计算加权求和
                weighted_tensors = [w * t for t, w in zip(highres_cam_norm, weights)]
                # torch.Size([2, 256, 256])
                highres_cam = torch.sum(torch.stack(weighted_tensors, 0), 0)[:, 0, :size[0], :size[1]]

                oo.append(highres_cam)
            highres_cam = [torch.unsqueeze(o, 1) for o in oo]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat - 1, "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    # torch.save(model.state_dict(), args.cam_weights_name + '222.pth', _use_new_zipfile_serialization=False)
    # sys.exit()
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
