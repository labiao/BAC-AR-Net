import sys

from .RepVGG import create_RepVGG_B1g2
from .wscodnet import *


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


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
                # print('change stride of ', n)

        self.label_enc = nn.Linear(1, 2048)
        astrous_rates = [6, 12, 18, 24]
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.1),
        #     _ASPP(in_ch=2048, out_ch=2, rates=astrous_rates)
        # )

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])

        self.pyramid_pooling = PyramidPooling(2048, 64)
        self.aspp = _ASPP(in_ch=2048, out_ch=64, rates=astrous_rates)
        self.conv1 = nn.ModuleList([
            basicConv(64, 64, k=1, s=1, p=0),
            basicConv(128, 64, k=1, s=1, p=0),
            basicConv(256, 64, k=1, s=1, p=0),
            basicConv(1024, 64, k=1, s=1, p=0),
            basicConv(2048, 64, k=1, s=1, p=0),
            # basicConv(2048, 2048, k=1, s=1, p=0)
        ])

        self.rfb = nn.ModuleList([
            RFB_modified(512, 64),
            RFB_modified(2048, 64)
        ])

        self.contrast = nn.ModuleList([
            Contrast_Block_Deep(64),
            Contrast_Block_Deep(64)
        ])

        self.fusion = nn.ModuleList([
            FFM(64),
            FFM(64),
            FFM(64),
            FFM(64)
        ])

        self.aggregation = nn.ModuleList([CAMM(64), CAMM(64)])

        self.head = nn.ModuleList([
            conv3x3(64, 2, bias=True),
            conv3x3(64, 2, bias=True),
            conv3x3(64, 2, bias=True),
            conv3x3(64, 2, bias=True),
            conv3x3(64, 2, bias=True),
        ])

        self.newly_added = nn.ModuleList([self.aspp, self.pyramid_pooling, self.conv1, self.rfb, self.contrast,
                                          self.fusion, self.aggregation, self.head, self.label_enc])

    def forward(self, img, label_cls):

        # y相当于权重作用于深层次特征了
        y = self.label_enc(label_cls).unsqueeze(-1).unsqueeze(-1)

        x0 = self.stage0(img)
        bk_stage2 = self.stage1(x0)
        bk_stage3 = self.stage2(bk_stage2)
        bk_stage4 = self.stage3(bk_stage3)
        bk_stage5 = self.stage4(bk_stage4)
        # torch.Size([16, 64, 128, 128])
        # torch.Size([16, 128, 64, 64])
        # torch.Size([16, 256, 32, 32])
        # torch.Size([16, 512, 16, 16])
        # torch.Size([16, 2048, 16, 16])
        # bk_stage5 = bk_stage5 * y
        # f_c3 = self.pyramid_pooling(bk_stage5)
        f_c3 = self.aspp(bk_stage5)
        shape = f_c3.size()[2:]
        # rfb实际上是LSR
        f_c2 = self.rfb[1](bk_stage5)
        fused3 = F.interpolate(f_c3, size=f_c2.size()[2:], mode='bilinear', align_corners=True)
        fused3 = self.fusion[2](f_c2, fused3)

        f_c1 = self.rfb[0](bk_stage4)
        fused2 = F.interpolate(f_c2, size=f_c1.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.fusion[1](f_c1, fused2)
        # contrast 实际上是 LCC
        f_t2 = self.conv1[2](bk_stage3)
        f_t2 = self.contrast[1](f_t2)

        a2 = F.interpolate(fused3, size=[f_t2.size(2) // 2, f_t2.size(3) // 2], mode='bilinear', align_corners=True)
        a2 = self.aggregation[1](a2, f_t2)

        f_t1 = self.conv1[1](bk_stage2)
        f_t1 = self.contrast[0](f_t1)

        a1 = F.interpolate(fused2, size=[f_t1.size(2) // 2, f_t1.size(3) // 2], mode='bilinear', align_corners=True)
        a1 = self.aggregation[0](a1, f_t1)

        a2 = F.interpolate(a2, size=a1.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.fusion[0](a1, a2)

        out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)

        out1 = F.interpolate(self.head[1](a1), size=shape, mode='bilinear', align_corners=False)
        # print(not x.isnan().any())
        out2 = F.interpolate(self.head[2](a2), size=shape, mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=False)
        out4 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=False)
        return out0, None, out1, out2, out3, out4

        # return logit, result, edge  # result代表融合结果

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, y):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        B, C, H, W = x.shape

        y = self.label_enc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        x = x * y

        logit = self.classifier(x)

        logit = (logit[0] + logit[1].flip(-1)) / 2

        return logit


class CAM2(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(CAM2, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x, y):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        B, C, H, W = x.shape

        y = self.label_enc(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        x = x * y
        # logit = self.pam(x)
        logit = self.classifier(x)

        logit = (logit[0] + logit[1].flip(-1)) / 2

        return logit


class CAM3(Net):

    def __init__(self, backbone_file, deploy=False, pretrained=True):
        super(CAM3, self).__init__(backbone_file, deploy=deploy, pretrained=pretrained)

    def forward(self, x, y):
        # y相当于权重作用于深层次特征了
        y = self.label_enc(y).unsqueeze(-1).unsqueeze(-1)

        x0 = self.stage0(x)
        bk_stage2 = self.stage1(x0)
        bk_stage3 = self.stage2(bk_stage2)
        bk_stage4 = self.stage3(bk_stage3)
        bk_stage5 = self.stage4(bk_stage4)
        # torch.Size([16, 64, 128, 128])
        # torch.Size([16, 128, 64, 64])
        # torch.Size([16, 256, 32, 32])
        # torch.Size([16, 512, 16, 16])
        # torch.Size([16, 2048, 16, 16])
        f_c3 = self.aspp(bk_stage5)
        shape = f_c3.size()[2:]
        # print(shape)
        # sys.exit(0)
        # rfb实际上是LSR
        f_c2 = self.rfb[1](bk_stage5)
        fused3 = F.interpolate(f_c3, size=f_c2.size()[2:], mode='bilinear', align_corners=True)
        fused3 = self.fusion[2](f_c2, fused3)

        f_c1 = self.rfb[0](bk_stage4)
        fused2 = F.interpolate(f_c2, size=f_c1.size()[2:], mode='bilinear', align_corners=True)
        fused2 = self.fusion[1](f_c1, fused2)
        # contrast 实际上是 LCC
        f_t2 = self.conv1[2](bk_stage3)
        f_t2 = self.contrast[1](f_t2)

        a2 = F.interpolate(fused3, size=[f_t2.size(2) // 2, f_t2.size(3) // 2], mode='bilinear', align_corners=True)
        a2 = self.aggregation[1](a2, f_t2)

        f_t1 = self.conv1[1](bk_stage2)
        f_t1 = self.contrast[0](f_t1)

        a1 = F.interpolate(fused2, size=[f_t1.size(2) // 2, f_t1.size(3) // 2], mode='bilinear', align_corners=True)
        a1 = self.aggregation[0](a1, f_t1)

        a2 = F.interpolate(a2, size=a1.size()[2:], mode='bilinear', align_corners=True)
        out0 = self.fusion[0](a1, a2)

        out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)

        # out1 = F.interpolate(self.head[1](a1), size=shape, mode='bilinear', align_corners=False)
        #
        # out2 = F.interpolate(self.head[2](a2), size=shape, mode='bilinear', align_corners=False)
        # out3 = F.interpolate(self.head[3](fused2), size=shape, mode='bilinear', align_corners=False)
        # out4 = F.interpolate(self.head[4](fused3), size=shape, mode='bilinear', align_corners=False)

        logit0 = (out0[0] + out0[1].flip(-1)) / 2
        return logit0
