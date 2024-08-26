import sys

import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
import importlib
import torch.nn.functional as F


def get_soft_label(input_tensor, num_class, data_type='float'):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    if (data_type == 'float'):
        output_tensor = output_tensor.float()
    elif (data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor


def boundaryLoss(pred, soft_y, theta0=3, theta=5):
    n, c, _, _ = pred.shape

    # softmax so that predicted map can be distributed in [0, 1]
    pred = torch.softmax(pred, dim=1)

    # one-hot vector of ground truth
    one_hot_gt = get_soft_label(soft_y.unsqueeze(1), 2).cuda()
    # print(one_hot_gt.shape)
    # boundary map
    gt_b = F.max_pool2d(
        1 - one_hot_gt[:, 1, :, :], kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b -= 1 - one_hot_gt[:, 1, :, :]

    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    # print(gt_b.shape, pred_b.shape, gt_b_ext.shape, pred_b_ext.shape)
    # torch.Size([32, 2, 256, 256])
    # torch.Size([32, 1, 64, 64])
    # torch.Size([32, 2, 256, 256])
    # torch.Size([32, 1, 64, 64])
    # reshape
    gt_b = gt_b.view(n, c, -1)
    pred_b = pred_b.view(n, c, -1)
    gt_b_ext = gt_b_ext.view(n, c, -1)
    pred_b_ext = pred_b_ext.view(n, c, -1)
    # sys.exit()
    # Precision, Recall
    P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
    R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

    # Boundary F1 Score
    BF1 = 2 * P * R / (P + R + 1e-7)

    # summing BF1 Score for each class and average over mini-batch
    loss = torch.mean(1 - BF1)

    return loss


def run(args):
    path_index = indexing.PathIndex(radius=5, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(
        path_index)

    train_dataset = voc12.dataloader.VOC12AffinityDataset(args.train_list,
                                                          label_dir=args.amn_ir_label_out_dir,
                                                          voc12_root=args.voc12_root,
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method="random",
                                                          rescale=(0.5, 1.5)
                                                          )
    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1 * args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].cuda(non_blocking=True)
            bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
            fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
            neg_label = pack['aff_neg_label'].cuda(non_blocking=True)
            label_amn = pack['reduced_label'].long().cuda(non_blocking=True)
            label_ = label_amn.clone()
            label_[label_amn == 255] = 0
            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss, edge_out = model(img, True)

            bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})
            # bl = boundaryLoss(edge_out, label_)

            total_loss = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (
                          avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'),
                          avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()

    infer_dataset = voc12.dataloader.VOC12ImageDataset(args.infer_list,
                                                       voc12_root=args.voc12_root,
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img'].cuda(non_blocking=True)

            aff, dp = model(img, False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.module.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
    print('done.')

    torch.save(model.module.state_dict(), args.irn_weights_name)
    torch.cuda.empty_cache()
