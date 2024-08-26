import os

import cv2
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader

import importlib

import voc12.dataloader
from misc import pyutils, imutils

from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm
from misc.lossutils import *


def run(args):
    model = getattr(importlib.import_module(args.amn_network), 'Net')(backbone_file=f"weights/{args.MODEL_WEIGHTS}")

    train_dataset = voc12.dataloader.VOC12SegmentationDataset(args.train_list,
                                                              label_dir=args.ir_label_out_dir,
                                                              voc12_root=args.voc12_root,
                                                              hor_flip=True,
                                                              crop_size=args.amn_crop_size,
                                                              crop_method="random",
                                                              rescale=(0.5, 1.5)
                                                              )

    train_data_loader = DataLoader(train_dataset, batch_size=args.amn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dataset = voc12.dataloader.VOC12SegmentationDataset(args.infer_list,
                                                            label_dir=args.ir_label_out_dir,
                                                            voc12_root=args.voc12_root,
                                                            crop_size=None,
                                                            crop_method="none",
                                                            )

    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    param_groups = model.trainable_parameters()

    optimizer = torch.optim.Adam(
        params=[
            {
                'params': param_groups[0],
                'lr': 5e-05,
                'weight_decay': 1.0e-4,
            },
            {
                'params': param_groups[1],
                'lr': 1e-03,
                'weight_decay': 1.0e-4,
            },
        ],
    )

    total_epochs = args.amn_num_epoches

    model = torch.nn.DataParallel(model).cuda()

    model.train()

    avg_meter = pyutils.AverageMeter()
    best_mIoU = 0.0
    for ep in range(total_epochs):
        loader_iter = iter(train_data_loader)

        pbar = tqdm(
            range(1, len(train_data_loader) + 1),
            total=len(train_data_loader),
            dynamic_ncols=True,
        )
        for iteration, _ in enumerate(pbar):
            optimizer.zero_grad()
            try:
                pack = next(loader_iter)
            except:
                loader_iter = iter(train_data_loader)
                pack = next(loader_iter)

            img = pack['img'].cuda(non_blocking=True)
            label_amn = pack['label'].long().cuda(non_blocking=True)
            label_cls = pack['label_cls'].cuda(non_blocking=True)
            # print(label_amn.shape)
            logit, sa_output = model(img, label_cls)

            B, C, H, W = logit.shape
            # print(sa_output.shape)  # 16 2 16 16
            label_amn = resize_labels(label_amn.cpu(), size=logit.shape[-2:]).cuda()

            label_ = label_amn.clone()
            label_[label_amn == 255] = 0

            given_labels = torch.full(size=(B, C, H, W), fill_value=args.eps / (C - 1)).cuda()

            given_labels.scatter_(dim=1, index=torch.unsqueeze(label_, dim=1), value=1 - args.eps)


            loss_pcl = balanced_cross_entropy(logit, label_amn, given_labels)

            consistency_multiscale = consistency_rotation = 0.0
            weight = nn.Parameter(torch.Tensor(3)).cuda(non_blocking=True)
            weight.data.fill_(1)
            if args.Lambda1 > 0.0:
                # print(args.Lambda1)
                # Multiscale consistency
                inputs_small = F.interpolate(img, scale_factor=args.scale_factor, mode='bilinear',
                                             align_corners=True,
                                             recompute_scale_factor=True)

                inputs_large = F.interpolate(img, scale_factor=args.scale_factor2, mode='bilinear',
                                             align_corners=True,
                                             recompute_scale_factor=True)

                # input to be scaled e.g 0.7
                pred2, _ = model(inputs_small, label_cls)
                pred2 = F.interpolate(pred2, size=(H, W), mode='bilinear', align_corners=True)

                # input to be scaled e.g 1.5
                pred3, _ = model(inputs_large, label_cls)
                pred3 = F.interpolate(pred3, size=(H, W), mode='bilinear', align_corners=True)
                consistency_multiscale = calc_jsd(
                    weight, label_amn, [logit, pred2, pred3], threshold=args.threshold)
            weight = nn.Parameter(torch.Tensor(4)).cuda(non_blocking=True)
            weight.data.fill_(1)
            if args.Lambda2 > 0.0:
                # print(args.Lambda2)
                inputs_rotate_90 = torch.rot90(img, k=-1, dims=[2, 3])  # 顺时针90
                inputs_rotate_fu90 = torch.rot90(img, k=1, dims=[2, 3])  # 逆时针90
                inputs_rotate_fu180 = torch.rot90(img, k=2, dims=[2, 3])  # 逆时针180
                _, sa_output1 = model(inputs_rotate_90, label_cls)
                _, sa_output2 = model(inputs_rotate_fu90, label_cls)
                _, sa_output3 = model(inputs_rotate_fu180, label_cls)

                sa_output1 = torch.rot90(sa_output1, k=1, dims=[2, 3])  # 顺时针90
                sa_output2 = torch.rot90(sa_output2, k=-1, dims=[2, 3])  # 逆时针90
                sa_output3 = torch.rot90(sa_output3, k=-2, dims=[2, 3])  # 逆时针180

                consistency_rotation = calc_jsd(
                    weight, label_amn, [sa_output, sa_output1, sa_output2, sa_output3], threshold=args.threshold)

            loss = loss_pcl + args.Lambda1 * consistency_multiscale

            loss.backward()

            optimizer.step()

            avg_meter.add({'loss': loss.item()})

            pbar.set_description(f"[{ep + 1}/{total_epochs}] "
                                 f"PCL: [{avg_meter.pop('loss'):.4f}]")

        with torch.no_grad():
            model.eval()
            labels = []
            preds = []

            for i, pack in enumerate(tqdm(val_data_loader)):
                img_name = pack['name'][0]
                img = pack['img']
                label_cls = pack['label_cls'][0]

                img = img.cuda()

                logit, _ = model(img, pack['label_cls'].cuda())

                size = img.shape[-2:]
                strided_up_size = imutils.get_strided_up_size(size, 16)

                valid_cat = torch.nonzero(label_cls)[:, 0]
                keys = np.pad(valid_cat + 1, (1, 0), mode='constant')

                logit_up = F.interpolate(logit, strided_up_size, mode='bilinear', align_corners=False)
                logit_up = logit_up[0, :, :size[0], :size[1]]

                logit_up = F.softmax(logit_up, dim=0)[keys].cpu().numpy()

                cls_labels = np.argmax(logit_up, axis=0)
                cls_labels = keys[cls_labels]

                preds.append(cls_labels.copy())

                img = cv2.imread(
                    os.path.join('Dataset/RRDSD/SegmentationClassAug', img_name + '.png'),
                    cv2.IMREAD_GRAYSCALE)

                gt_label = img // 255

                labels.append(gt_label.copy())

            confusion = calc_semantic_segmentation_confusion(preds, labels)

            gtj = confusion.sum(axis=1)
            resj = confusion.sum(axis=0)
            gtjresj = np.diag(confusion)
            denominator = gtj + resj - gtjresj
            iou = gtjresj / denominator

            print(f'[{ep + 1}/{total_epochs}] miou: {np.nanmean(iou):.4f}')

            if np.nanmean(iou) > best_mIoU:
                best_mIoU = np.nanmean(iou)
                torch.save(model.module.state_dict(), args.amn_weights_name + '.pth')
                print(best_mIoU)

            model.train()

    # torch.save(model.module.state_dict(), args.amn_weights_name + '.pth')
    torch.cuda.empty_cache()
