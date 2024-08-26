import sys
from .MSSA import BiGFF
import torch
import torch.nn as nn
import torch.nn.functional as F
from .RepVGG import create_RepVGG_B1g2


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

        astrous_rates = [6, 12, 18, 24]

        self.label_enc = nn.Linear(1, 2048)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            _ASPP(in_ch=2048, out_ch=2, rates=astrous_rates)
        )
        # self.pam = DANetHead(2048, 2, nn.BatchNorm2d)
        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])

        self.newly_added = nn.ModuleList([self.classifier, self.label_enc])

    def forward(self, img, label_cls):
        # y相当于权重作用于深层次特征了
        y = self.label_enc(label_cls).unsqueeze(-1).unsqueeze(-1)

        x = self.stage0(img)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # torch.Size([16, 64, 128, 128])
        # torch.Size([16, 128, 64, 64])
        # torch.Size([16, 256, 32, 32])
        # torch.Size([16, 512, 16, 16])
        # torch.Size([16, 2048, 16, 16])
        x = x * y
        # sa_output = None
        # sa_output = self.pam(x)
        # EdgeNet Block########################
        # up1 = F.interpolate(x3, size=[64, 64], mode="bilinear")  # 64*64*512
        # up1 = self.b_conv1(up1)  # 64*64*128
        # up_concat = torch.cat([up1, x1], dim=1)  # 64*64*256
        #
        # up2 = self.b_conv2(up_concat)  # 64*64*64
        #
        # edge_map = F.interpolate(up2, size=[256, 256], mode="bilinear")  # 256*256*64
        # up2 = F.interpolate(up2, size=[16, 16], mode="bilinear")  # 16*16*64
        # edge = self.b_conv3(edge_map)  # 256*256*1
        logit = self.classifier(x)
        # # sa_output = F.interpolate(sa_output, size=[64, 64], mode="bilinear")  # 64*64*64
        # # gated fusion##########################
        # atten_concat = self.bigff(sa_output, up2)  # 16*16*128
        # result = self.b_conv4(atten_concat)  # 256*256*2
        # return logit, sa_output  # result代表融合结果
        return logit, None  # result代表融合结果

    # def train(self, mode=True):
    #     for p in self.resnet50.conv1.parameters():
    #         p.requires_grad = False
    #     for p in self.resnet50.bn1.parameters():
    #         p.requires_grad = False

    def trainable_parameters(self):

        return list(self.backbone.parameters()), list(self.newly_added.parameters())


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        # self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        # self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        #
        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # feat2 = self.conv5c(x)
        # sc_feat = self.sc(feat2)
        # sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        # feat_sum = sa_conv+sc_conv

        # sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        # return tuple(output)

        return sa_output


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
