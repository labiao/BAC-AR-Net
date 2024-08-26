import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from PIL import Image


def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class = list(x.size())[1]
    if (tensor_dim == 5):
        x_perm = x.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        x_perm = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    y = torch.reshape(x_perm, (-1, num_class))
    return y


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


def lace(predict, soft_y, priors=[0.7, 0.3], tau=1.0, cls_num=2):
    predict = reshape_tensor_to_2D(predict)
    log_prior = torch.tensor(np.log(np.array(priors) + 1e-8)).cuda()
    predict = predict + tau * log_prior.unsqueeze(0)

    predict = nn.Softmax(dim=1)(predict)
    soft_y = get_soft_label(soft_y.unsqueeze(1), cls_num).cuda()

    soft_y = reshape_tensor_to_2D(soft_y)

    ce = - soft_y * torch.log(predict)
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce)

    return ce


def calc_jsd(weight, labels1_a, predList, threshold=0.8, Mask_label255_sign='no'):
    Mask_label255 = (labels1_a < 255).float()  # do not compute the area that is irrelavant (dataaug)  b,h,w
    weight_softmax = F.softmax(weight, dim=0)
    probs = [F.softmax(logits, dim=1) for i, logits in enumerate(predList)]

    weighted_probs = [weight_softmax[i] * prob for i, prob in enumerate(probs)]  # weight_softmax[i]*
    mixture_label = (torch.stack(weighted_probs)).sum(axis=0)
    # mixture_label = torch.clamp(mixture_label, 1e-7, 1)  # h,c,h,w
    mixture_label = torch.clamp(mixture_label, 1e-3, 1 - 1e-3)  # h,c,h,w

    # add this code block for early torch version where torch.amax is not available
    if torch.__version__ == "1.5.0" or torch.__version__ == "1.6.0":
        _, max_probs = torch.max(mixture_label * Mask_label255.unsqueeze(1), dim=-3, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-2, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-1, keepdim=True)
    else:
        max_probs = torch.amax(mixture_label * Mask_label255.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    mask = max_probs.ge(threshold).float()

    logp_mixture = mixture_label.log()

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    if Mask_label255_sign == 'yes':
        consistency = sum(log_probs) * Mask_label255
    else:
        consistency = sum(log_probs)

    return torch.mean(consistency)


def balanced_cross_entropy(logits, labels, one_hot_labels):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """

    N, C, H, W = logits.shape

    assert one_hot_labels.size(0) == N and one_hot_labels.size(
        1) == C, f'label tensor shape is {one_hot_labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    loss_structure = -torch.sum(log_logits * one_hot_labels, dim=1)  # (N)
    # print(log_logits.shape, loss_structure.shape)
    # torch.Size([16, 2, 16, 16]) torch.Size([16, 16, 16])

    ignore_mask_bg = torch.zeros_like(labels)
    ignore_mask_fg = torch.zeros_like(labels)

    ignore_mask_bg[labels == 0] = 1
    ignore_mask_fg[(labels != 0) & (labels != 255)] = 1
    # print(labels.shape, ignore_mask_bg.shape)
    loss_bg = (loss_structure * ignore_mask_bg).sum() / ignore_mask_bg.sum()
    loss_fg = (loss_structure * ignore_mask_fg).sum() / ignore_mask_fg.sum()

    return (loss_bg + loss_fg) / 2


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


"""
The following code is modified from the file in https://github.com/siyueyu/SCWSSOD/blob/f8650567cbbc8df5bf6edc32a633c47a885574cd/lscloss.py.
Credit for them.
"""
class FeatureLoss(torch.nn.Module):
    """
    This loss function based on the following paper.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded)

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))


        out = {
            'loss': loss.mean(),
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = FeatureLoss._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    # feature = LocalSaliencyCoherence._downsample(
                    #     feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    # )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * FeatureLoss._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = FeatureLoss._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis
