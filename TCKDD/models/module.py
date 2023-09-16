import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# utils and pooling, upsampling

def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    """Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))
    if index.ndim > 1: # recursive implementation
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)
    return output


def gather(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    # x: [n_points+1, f_dim]
    # idx: [n_points, n_neighbors](pooling) or [n_points](upsampling)
    for i, ni in enumerate(idx.size()[1:]):
        x = x.unsqueeze(i+1)
        new_s = list(x.size())
        new_s[i+1] = ni
        x = x.expand(new_s)
    n = len(idx.size())
    for i, di in enumerate(x.size()[n:]):
        idx = idx.unsqueeze(i+n)
        new_s = list(idx.size())
        new_s[i+n] = di
        idx = idx.expand(new_s)
    return x.gather(0, idx)


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors.
    WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    return gather(x, inds[:, 0]) # [n2, d]


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    pool_features = gather(x, inds) # n2, max_num, d]
    max_features, _ = torch.max(pool_features, 1) # [n2, d]
    return max_features


class NearestUpsampleBlock(nn.Module):
    def __init__(self):
        super(NearestUpsampleBlock, self).__init__()
    def forward(self, x, upsamples):
        return closest_pool(x, upsamples)

class MaxPoolBlock(nn.Module):
    def __init__(self):
        super(MaxPoolBlock, self).__init__()
    def forward(self, x, pools):
        return max_pool(x, pools)


# normalization

class AddBias(nn.Module):
    def __init__(self, in_channels):
        super(AddBias, self).__init__()
        self.bias = Parameter(torch.zeros(in_channels).float(), requires_grad=True)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        return x + self.bias


class BatchNorm(nn.Module):
    def __init__(self, in_channels, momentum=None):
        super(BatchNorm, self).__init__()
        if momentum is None: self.batch_norm = nn.BatchNorm1d(in_channels)
        else: self.batch_norm = nn.BatchNorm1d(in_channels, momentum=momentum)

    def forward(self, x:torch.FloatTensor):
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze()


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x:torch.FloatTensor):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x.squeeze()

    def __repr__(self):
        return self.norm.__repr__()


# fundamental blocks

class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, group_norm, bn_momentum=None, has_relu=True):
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm if norm is not None else 'add_bias'
        self.has_relu = has_relu

        # self.mlp = nn.Conv1d(in_dim, out_dim, 1, 1)
        self.mlp = nn.Linear(in_channels, out_channels, bias=True)
        if norm is None: # just add a learnable bias
            self.norm = AddBias(out_channels)
        elif norm=='batch_norm':
            self.norm = BatchNorm(out_channels, bn_momentum)
        elif norm=='layer_norm':
            self.norm = nn.LayerNorm(out_channels)
        else: self.norm = GroupNorm(group_norm, out_channels)
        if has_relu: self.leaky_relu = nn.LeakyReLU(0.1)
        else: self.leaky_relu = None

    def forward(self, x:torch.FloatTensor):
        # x = self.mlp(x.unsqueeze(0)).squeeze(0)
        x = self.mlp(x)
        x = self.norm(x)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_channels: {:d}, out_channels: {:d}, norm: {:s}, ReLU: {:s})'\
            .format(self.in_channels, self.out_channels, self.norm_type, str(self.has_relu))


class LastUnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(LastUnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        return self.mlp(x)
    
    def __repr__(self):
        return 'LastUnaryBlock(in_channels: {:d}, out_channels: {:d})'\
            .format(self.in_channels, self.out_channels)
