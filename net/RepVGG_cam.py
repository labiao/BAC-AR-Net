import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils

from .RepVGG import create_RepVGG_B1g2


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

        self.classifier = nn.Conv2d(2048, 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        # print(x.shape) # [16, 256, 32, 32]
        x = self.stage3(x)
        # print(x.shape) # [16, 512, 16, 16]
        x = self.stage4(x)
        # print(x.shape) # [16, 2048, 16, 16]
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 1)

        return x

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class Net_CAM(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net_CAM, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 1)  # 1个类别

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)

        return x, cams, feature


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


class Net_Feature(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(Net_Feature, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        return feature


class CAM2(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(CAM2, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x, separate=False):
        x = self.stage0(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


class GradCAMPlusPlus(Net):
    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(GradCAMPlusPlus, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x, separate=False):
        x = self.stage0(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x_logit = torchutils.gap2d(x, keepdims=True)
        x_logit = self.classifier(x_logit)
        x_logit = x_logit.view(-1, 1)

        # x = F.conv2d(x, self.classifier.weight)
        # if separate:
        #     return x
        # x = F.relu(x)
        #
        # x = x[0] + x[1].flip(-1)
        self.output = x_logit
        return x_logit

    def backward(self, index):
        self.zero_grad()
        output_scalar = self.output[index]  # 选择当前样本的输出
        output_scalar.backward(retain_graph=True)

    def generate_cam(self, index, feature_maps, gradients):
        weights = []
        for i in range(len(feature_maps)):
            # print(i, gradients[i].shape, gradients[i][index].shape)
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

        # cam -= cam.min()
        # cam /= cam.max()
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
