import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class CGUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CGUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Net(nn.Module):

    def __init__(self, stride=32, n_classes=2):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool,
                                        self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool,
                                        self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.cg1 = CGUnit(256, 256)
        self.cg2 = CGUnit(512, 512)
        self.cg3 = CGUnit(1024, 1024)
        self.cg4 = CGUnit(2048, 2048)

        self.classifier1 = nn.Conv2d(256, self.n_classes, 1, bias=False)
        self.classifier2 = nn.Conv2d(512, self.n_classes, 1, bias=False)
        self.classifier3 = nn.Conv2d(1024, self.n_classes, 1, bias=False)
        self.classifier4 = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.cg1, self.cg2, self.cg3, self.cg4,
                                          self.classifier1, self.classifier2, self.classifier3, self.classifier4])

    def forward(self, x):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        # print(x.shape) # [16, 512, 32, 32]
        x3 = self.stage3(x2)
        # print(x.shape) # [16, 1024, 16, 16]
        x4 = self.stage4(x3)
        # print(x1.shape) # [16, 2048, 8, 8]

        x1 = self.cg1(x1)
        x2 = self.cg2(x2)
        x3 = self.cg3(x3)
        x4 = self.cg4(x4)

        x1 = torchutils.gap2d(x1, keepdims=True)
        x2 = torchutils.gap2d(x2, keepdims=True)
        x3 = torchutils.gap2d(x3, keepdims=True)
        x4 = torchutils.gap2d(x4, keepdims=True)

        x1 = self.classifier1(x1).view(-1, self.n_classes)
        x2 = self.classifier2(x2).view(-1, self.n_classes)
        x3 = self.classifier3(x3).view(-1, self.n_classes)
        x4 = self.classifier4(x4).view(-1, self.n_classes)

        return x1, x2, x3, x4

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class GradCAMPlusPlus(Net):
    def __init__(self):
        super(GradCAMPlusPlus, self).__init__()
        self.output = []

    def forward(self, x, separate=False):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x1 = self.cg1(x1)
        x2 = self.cg2(x2)
        x3 = self.cg3(x3)
        x4 = self.cg4(x4)

        x1 = torchutils.gap2d(x1, keepdims=True)
        x2 = torchutils.gap2d(x2, keepdims=True)
        x3 = torchutils.gap2d(x3, keepdims=True)
        x4 = torchutils.gap2d(x4, keepdims=True)

        x1 = self.classifier1(x1).view(-1, 1)
        x2 = self.classifier2(x2).view(-1, 1)
        x3 = self.classifier3(x3).view(-1, 1)
        x4 = self.classifier4(x4).view(-1, 1)

        self.output = x1 + x2 + x3 + x4
        return x1, x2, x3, x4

    def backward(self, index):
        self.zero_grad()
        # print(self.output, index2, index)
        output_scalar = self.output[index]
        output_scalar.backward(retain_graph=True)

    def generate_cam(self, index, feature_maps, gradients):
        weights = []
        for i in range(len(feature_maps)):
            # print(i, gradients[i].shape, feature_maps[i].shape, gradients[i][index].shape)
            grad = gradients[i][index]  # 针对当前样本的梯度
            fm = feature_maps[i][index]
            alpha_num = grad.pow(2)
            alpha_denom = 2 * grad.pow(2) + fm.mul(grad.pow(2)).sum((1, 2), keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom + 1e-7)

            positive_grad = F.relu(grad)
            weight = (alpha * positive_grad).sum((1, 2), keepdim=True)
            weights.append(weight)

        cam = torch.zeros_like(feature_maps[0][index][0])
        for i, feature_map in enumerate(feature_maps):
            cam += (feature_map[index] * weights[i]).sum(0)
        cam = F.relu(cam)

        return cam.unsqueeze(0)


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, separate=False):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x1 = self.cg1(x1)
        x2 = self.cg2(x2)
        x3 = self.cg3(x3)
        x4 = self.cg4(x4)

        x1 = F.conv2d(x1, self.classifier1.weight)
        x2 = F.conv2d(x2, self.classifier2.weight)
        x3 = F.conv2d(x3, self.classifier3.weight)
        x4 = F.conv2d(x4, self.classifier4.weight)

        if separate:
            return x1, x2, x3, x4
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)

        x1 = x1[0] + x1[1].flip(-1)
        x2 = x2[0] + x2[1].flip(-1)
        x3 = x3[0] + x3[1].flip(-1)
        x4 = x4[0] + x4[1].flip(-1)

        return x1, x2, x3, x4

