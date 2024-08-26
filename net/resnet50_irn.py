import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50
from my_functionals import GatedSpatialConv as gsc

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # backbone
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.mean_shift = Net.MeanShift(2)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # branch: displacement field
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            self.mean_shift
        )

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(1024, 1, 1)
        self.res1 = resnet50.BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv2d(64, 32, 1)
        self.res2 = resnet50.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = resnet50.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList(
            [self.dsn3, self.dsn4, self.dsn5, self.res1, self.d1, self.res2, self.d2, self.res3, self.d3, self.fuse, self.gate1, self.gate2, self.gate3])
        # self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.dp_layers = nn.ModuleList([self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7])

    class MeanShift(nn.Module):

        def __init__(self, num_features):
            super(Net.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        x_size = x2.size()
        s3 = F.interpolate(self.dsn3(x2), x_size[2:],
                           mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(x3), x_size[2:],
                           mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.dsn5(x4), x_size[2:],
                           mode='bilinear', align_corners=True)

        # m1f = F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True)

        cs = self.res1(x1)  # 残差块
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        # self.d1 = nn.Conv2d(64, 32, 1)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)  # cs是32通道（64——32） s3是一个通道
        cs = self.res2(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)  # cs是16通道（32——16） s4是一个通道
        cs = self.res3(cs)
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        cs = self.gate3(cs, s5)  # # cs是16通道（16——8） s5是一个通道
        # self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
        cs = self.fuse(cs)
        edge_out = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)

        # edge1 = self.fc_edge1(x1)
        # edge2 = self.fc_edge2(x2)
        # edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        # edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        # edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        # edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def trainable_parameters(self):
        return (tuple(self.edge_layers.parameters()),
                tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class AffinityDisplacementLoss(Net):

    path_indices_prefix = "path_indices"

    def __init__(self, path_index):

        super(AffinityDisplacementLoss, self).__init__()

        self.path_index = path_index

        self.n_path_lengths = len(path_index.path_indices)
        for i, pi in enumerate(path_index.path_indices):
            self.register_buffer(AffinityDisplacementLoss.path_indices_prefix + str(i), torch.from_numpy(pi))

        self.register_buffer(
            'disp_target',
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(path_index.search_dst).transpose(1, 0), 0), -1).float())

    # 对应论文中公式(7)
    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_lengths):
            ind = self._buffers[AffinityDisplacementLoss.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)

        return aff_cat

    def to_pair_displacement(self, disp):
        height, width = disp.size(2), disp.size(3)
        radius_floor = self.path_index.radius_floor

        cropped_height = height - radius_floor
        cropped_width = width - 2 * radius_floor

        disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]

        disp_dst = [disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
                       for dy, dx in self.path_index.search_dst]
        disp_dst = torch.stack(disp_dst, 2)

        pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst
        pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1), pair_disp.size(2), -1)

        return pair_disp

    def to_displacement_loss(self, pair_disp):
        return torch.abs(pair_disp - self.disp_target)

    def forward(self, *inputs):
        x, return_loss = inputs
        edge_out, dp_out = super().forward(x)

        if return_loss is False:
            return edge_out, dp_out

        aff = self.to_affinity(torch.sigmoid(edge_out))
        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        pair_disp = self.to_pair_displacement(dp_out)
        dp_fg_loss = self.to_displacement_loss(pair_disp)
        dp_bg_loss = torch.abs(pair_disp)

        return pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss, edge_out


class EdgeDisplacement(Net):

    def __init__(self, crop_size=256, stride=4):
        super(EdgeDisplacement, self).__init__()
        self.crop_size = crop_size
        self.stride = stride

    def forward(self, x):
        feat_size = (x.size(2)-1)//self.stride+1, (x.size(3)-1)//self.stride+1

        x = F.pad(x, [0, self.crop_size-x.size(3), 0, self.crop_size-x.size(2)])
        edge_out, dp_out = super().forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        dp_out = dp_out[..., :feat_size[0], :feat_size[1]]

        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        dp_out = dp_out[0]

        return edge_out, dp_out


