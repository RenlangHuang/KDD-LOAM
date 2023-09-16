import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from kernels.kernel_points import load_kernels
from models.module import *


class KPConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, p_dim=3):
        """
        Initialize parameters for KPConv. Deformable KPConv is not supported!
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param kernel_size: Number of kernel points.
        :param radius: radius used for kernel point init.
        :param sigma: influence radius of each kernel point.
        :param p_dim: dimension of the point space.
        """
        super(KPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma

        self.weights = Parameter(torch.zeros((kernel_size, in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Initialize the kernel point positions in a sphere
        K_points_numpy = load_kernels(self.radius, kernel_size, p_dim, 'center')
        self.kernel_points = Parameter(torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False)


    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :]) + 1e6), 0)
        neighbors = s_points[neighbor_indices, :] - q_points.unsqueeze(1) # [B, N, 3]
        differences = neighbors.unsqueeze(2) - self.kernel_points # [B, N, M, 3]
        sq_distances = torch.sum(differences ** 2, dim=3) # [B, N, M]

        # get kernel point influences [points, kpoints, neighbors]
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)
        all_weights = torch.transpose(all_weights, 1, 2) # [B, M, N]

        s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)
        neighb_x = gather(s_feats, neighbor_indices) # [B, N, Cin]
        weighted_features = torch.matmul(all_weights, neighb_x) # [B, M, Cin]
        weighted_features = weighted_features.permute((1, 0, 2)) # [M, B, Cin]
        kernel_outputs = torch.matmul(weighted_features, self.weights)
        output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

        # normalization term
        neighbor_features_sum = torch.sum(neighb_x, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        return output_features / neighbor_num.unsqueeze(1)

    def __repr__(self):
        return 'KPConv(in_channels: {:d}, out_channels: {:d}, radius: {:.2f}, sigma: {:.2f})'\
            .format(self.in_channels, self.out_channels, self.radius, self.sigma)


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, norm, group_norm, bn_momentum=None):
        super(SimpleConvBlock, self).__init__()
        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma)
        if norm is None: # just add a learnable bias
            self.norm = AddBias(out_channels)
        elif norm=='batch_norm':
            self.norm = BatchNorm(out_channels, bn_momentum)
        elif norm=='layer_norm':
            self.norm = nn.LayerNorm(out_channels)
        else: self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        return self.leaky_relu(self.norm(x))


class ResidualConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, radius, sigma, norm, group_norm, bn_momentum=None, strided=False):
        super(ResidualConvBlock, self).__init__()
        self.strided = strided

        if in_channels != out_channels//4:
            self.unary1 = UnaryBlock(in_channels, out_channels//4, norm, group_norm, bn_momentum)
        else: self.unary1 = nn.Identity()
        
        self.set_abstraction = KPConv(out_channels//4, out_channels//4, kernel_size, radius, sigma)
        if norm is None: # just add a learnable bias
            self.norm = AddBias(out_channels//4)
        elif norm=='batch_norm':
            self.norm = BatchNorm(out_channels//4, bn_momentum)
        elif norm=='layer_norm':
            self.norm = nn.LayerNorm(out_channels//4)
        else: self.norm = GroupNorm(group_norm, out_channels//4)
        self.unary2 = UnaryBlock(out_channels//4, out_channels, norm, group_norm, bn_momentum, False)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, norm, group_norm, bn_momentum, False)
        else: self.unary_shortcut = nn.Identity()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):        
        x = self.unary1(s_feats)
        x = self.set_abstraction(x, q_points, s_points, neighbor_indices)
        x = self.leaky_relu(self.norm(x))
        x = self.unary2(x)
        if self.strided:
            shortcut = max_pool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)
        return self.leaky_relu(x + shortcut)


class KPFCNN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, norm, group_norm=32):
        super(KPFCNN, self).__init__() # Deformable KPConv is not supported!

        self.encoder1_1 = SimpleConvBlock(
            input_dim, init_dim, kernel_size, init_radius, init_sigma, norm, group_norm
        )
        self.encoder1_2 = ResidualConvBlock(
            init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, norm, group_norm
        )

        self.encoder2_1 = ResidualConvBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, norm, group_norm, strided=True
        )
        self.encoder2_2 = ResidualConvBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, norm, group_norm
        )
        self.encoder2_3 = ResidualConvBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, norm, group_norm
        )

        self.encoder3_1 = ResidualConvBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, norm, group_norm, strided=True
        )
        self.encoder3_2 = ResidualConvBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, norm, group_norm
        )
        self.encoder3_3 = ResidualConvBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, norm, group_norm
        )

        self.encoder4_1 = ResidualConvBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, norm, group_norm, strided=True
        )
        self.encoder4_2 = ResidualConvBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, norm, group_norm
        )
        self.encoder4_3 = ResidualConvBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, norm, group_norm
        )

        self.encoder5_1 = ResidualConvBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, norm, group_norm, strided=True
        )
        self.encoder5_2 = ResidualConvBlock(
            init_dim * 16, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, norm, group_norm
        )
        self.encoder5_3 = ResidualConvBlock(
            init_dim * 32, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, norm, group_norm
        )

        self.decoder4 = UnaryBlock(init_dim * 48, init_dim * 16, norm, group_norm)
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, norm, group_norm)
        self.decoder2 = UnaryBlock(init_dim * 12, init_dim * 4, norm, group_norm)
        self.decoder1 = UnaryBlock(init_dim * 6, init_dim * 2, norm, group_norm)
        self.nearest_upsample = NearestUpsampleBlock()

        self.description_head = LastUnaryBlock(init_dim * 2, output_dim)
        self.detection_head = nn.Sequential(
            nn.Linear(init_dim * 2, init_dim), nn.Softplus(),
            nn.Linear(init_dim, init_dim//2), nn.Softplus(),
            nn.Linear(init_dim//2, 1, bias=False), nn.Softplus()
        )
        print(self)

    def forward(self, data_dict):
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = data_dict['features']
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        feature = self.nearest_upsample(feats_s5, upsampling_list[3])
        feature = torch.cat([feature, feats_s4], dim=1)
        feature = self.decoder4(feature)
        
        feature = self.nearest_upsample(feature, upsampling_list[2])
        feature = torch.cat([feature, feats_s3], dim=1)
        feature = self.decoder3(feature)

        feature = self.nearest_upsample(feature, upsampling_list[1])
        feature = torch.cat([feature, feats_s2], dim=1)
        feature = self.decoder2(feature)

        feature = self.nearest_upsample(feature, upsampling_list[0])
        feature = torch.cat([feature, feats_s1], dim=1)
        feature = self.decoder1(feature)

        descriptor = self.description_head(feature)
        saliency = self.detection_head(feature)
        return F.normalize(descriptor, p=2, dim=-1), saliency
