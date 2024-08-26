import os
import numpy as np
import imageio
from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
from tqdm import tqdm
from PIL import Image

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


def _work(process_id, infer_dataset, args):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(tqdm(infer_data_loader, position=process_id, desc=f'[PID{process_id}]')):
        # img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        img_name = pack['name'][0]
        img = pack['img'][0].numpy()
        if not os.path.exists(os.path.join(args.amn_cam_out_dir, img_name + '.npy')):
            continue
        cam_dict = np.load(os.path.join(args.amn_cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']

        cams = cams[1:, ...]

        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        if keys.shape[0] == 1:
            imageio.imwrite(os.path.join(args.amn_ir_label_out_dir, img_name + '.png'),
                            np.zeros((img.shape[:2])).astype(np.uint8))
            continue
        # print(keys, keys.shape[0]) # [0] 1 [0 1] 2
        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        out = Image.fromarray(conf.astype(np.uint8), mode='P')
        out.putpalette(palette)
        out.save(os.path.join(os.path.join(args.amn_ir_label_out_dir, img_name + '_palette.png')))

        imageio.imwrite(os.path.join(args.amn_ir_label_out_dir, img_name + '.png'),
                        conf.astype(np.uint8))


def run(args):
    dataset = voc12.dataloader.VOC12ImageDataset(args.train_list, voc12_root=args.voc12_root, img_normal=None,
                                                 to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)
    # print(args.num_workers)
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
