import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils

from .RepVGG import create_RepVGG_B1g2


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

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net, self).__init__()
        self.repvgg = create_RepVGG_B1g2(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            self.repvgg.load_state_dict(ckpt)

        self.stage0, self.stage1, self.stage2, self.stage3, \
            self.stage4 = self.repvgg.stage0, self.repvgg.stage1, \
            self.repvgg.stage2, self.repvgg.stage3, self.repvgg.stage4

        for n, m in self.stage4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                m.dilation, m.padding, m.stride = (1, 1), (1, 1), (1, 1)
                # print(m.dilation, m.padding, m.stride, m.kernel_size)  # (1, 1) (1, 1) (2, 2)
                # print('change dilation, padding, stride of ', n)
            elif 'rbr_1x1' in n and isinstance(m, nn.Conv2d):
                m.stride = (1, 1)
                print('change stride of ', n)
        self.cg1 = CGUnit(128, 128)
        self.cg2 = CGUnit(256, 256)
        self.cg3 = CGUnit(512, 512)
        self.cg4 = CGUnit(2048, 2048)

        self.classifier1 = nn.Conv2d(128, 1, 1, bias=False)
        self.classifier2 = nn.Conv2d(256, 1, 1, bias=False)
        self.classifier3 = nn.Conv2d(512, 1, 1, bias=False)
        self.classifier4 = nn.Conv2d(2048, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.cg1, self.cg2, self.cg3, self.cg4,
                                          self.classifier1, self.classifier2, self.classifier3, self.classifier4])

    def forward(self, x):

        x = self.stage0(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        # print(x.shape) # [16, 256, 32, 32]
        x3 = self.stage3(x2)
        # print(x.shape) # [16, 512, 16, 16]
        x4 = self.stage4(x3)
        # print(x1.shape) # [16, 2048, 16, 16]

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

        return x1, x2, x3, x4

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class GradCAMPlusPlus(Net):
    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(GradCAMPlusPlus, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)
        self.output = []

    def forward(self, x, separate=False):
        x = self.stage0(x)
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
            alpha_num = grad.pow(2)
            alpha_denom = 2 * grad.pow(2) + grad.sum((1, 2), keepdim=True).pow(2) + 1e-7
            alpha = alpha_num / alpha_denom

            positive_grad = F.relu(grad)
            weight = (alpha * positive_grad).sum((1, 2), keepdim=True)
            weights.append(weight)

        cam = torch.zeros_like(feature_maps[0][index][0])
        for i, feature_map in enumerate(feature_maps):
            cam += (feature_map[index] * weights[i]).sum(0)
        cam = F.relu(cam)

        return cam.unsqueeze(0)


if __name__ == '__main__':
    feature_maps = []
    gradients = []


    def save_feature_map(module, input, output):
        feature_maps.append(output)


    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])


    # 使用示例
    model = GradCAMPlusPlus(None, deploy=True, pretrained=False)
    input_tensor = torch.randn(2, 3, 256, 256)  # 示例输入
    model.stage4.register_forward_hook(save_feature_map)
    model.stage4.register_backward_hook(save_gradient)
    output = model(input_tensor)
    # 对于批次中的每个样本，计算梯度并生成CAM
    cams = []
    for i in range(input_tensor.size(0)):
        model.backward(i)
        cam = model.generate_cam(i)
        cams.append(cam)
    cam = cams[0] + cams[1].flip(-1)
