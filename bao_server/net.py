import torch.nn as nn
import torch
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees
import numpy as np

NUM_OTHER_HUB = 1
NUM_HASHJOIN_HUB = 2
NUM_NESTEDLOOP_HUB = 3

def left_child(x):
    if x is None or not (isinstance(x, tuple) and len(x)==3):
        return None
    return x[1]

def right_child(x):
    if x is None or not (isinstance(x, tuple) and len(x)==3):
        return None
    return x[2]

def features(x):
    if x is None:
        # 返回一个默认的零向量，例如长度为1的零数组
        return np.zeros(1)
    return x[0]


class AttentionMerger(nn.Module):
    def __init__(self, input_dim, input_num):
        """
        Args:
            input_dim: 每个分支输出的特征维度（这里不直接用，但可以用于扩展）
            input_num: 输入分支的数量（这里为 3）
        """
        super(AttentionMerger, self).__init__()
        # 初始化注意力权重参数，形状为 [input_num, 1]
        self.attention_weights = nn.Parameter(torch.randn(input_num, 1))
        # 对注意力权重做 softmax，保证各分支的权重和为1
        self.softmax = nn.Softmax(dim=0)

    def forward(self, conv_outputs):
        """
        Args:
            conv_outputs: list of tensors from the 3 conv layers, each with shape [batch_size, feature_dim]
        Returns:
            weighted_sum: 融合后的输出，形状为 [batch_size, feature_dim]
        """
        # 将三个输出堆叠，得到形状 [3, batch_size, feature_dim]
        stacked_outputs = torch.stack(conv_outputs, dim=0)
        # 对注意力权重做 softmax，得到 [3, 1]
        weights = self.softmax(self.attention_weights)
        # 调整权重形状以便广播： [3, 1, 1]
        weights = weights.view(-1, 1, 1)
        # 利用广播机制，对每个分支的每个 feature 进行加权
        weighted_outputs = weights * stacked_outputs
        # 在第0个维度上求和，融合三个分支，结果形状为 [batch_size, feature_dim]
        weighted_sum = torch.sum(weighted_outputs, dim=0)
        return weighted_sum

class BaoNet(nn.Module):
    def __init__(self, in_channels):
        super(BaoNet, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = False
        
        self.conv_module_list_other = nn.ModuleList()
        self.conv_module_list_hash_join = nn.ModuleList()
        self.conv_module_list_nested_loop_join = nn.ModuleList()
        
        self.attention_merger = AttentionMerger(64, 3)
        
        for i in range(NUM_OTHER_HUB):
            self.conv_module_list_other.append(
                nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        ))
            
        for i in range(NUM_HASHJOIN_HUB):
            self.conv_module_list_hash_join.append(
                nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        ))
        for i in range(NUM_NESTEDLOOP_HUB):
            self.conv_module_list_nested_loop_join.append(
                nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        ))
        
        self.out_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def in_channels(self):
        return self.__in_channels
        
    def forward_new(self, a,b,c):
        trees_other = prepare_trees(a, features, left_child, right_child,
                              cuda=self.__cuda)
        trees_hash_join = prepare_trees(b, features, left_child, right_child,
                                cuda=self.__cuda)
        trees_nested_loop_join = prepare_trees(c, features, left_child, right_child,
                                        cuda=self.__cuda)
        # FIXME hardcoded now
        after_conv_other = self.conv_module_list_other[0](trees_other)
        after_conv_hash_join = self.conv_module_list_hash_join[0](trees_hash_join)
        after_conv_nested_loop_join = self.conv_module_list_nested_loop_join[0](trees_nested_loop_join)
        after_conv_output = [after_conv_other, after_conv_hash_join, after_conv_nested_loop_join]
        after_conv_final = self.attention_merger(after_conv_output)
        out = self.out_mlp(after_conv_final)
        return out
    
    def forward(self, x):
        trees_other = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        trees_hash_join = prepare_trees(x, features, left_child, right_child,
                                cuda=self.__cuda)
        trees_nested_loop_join = prepare_trees(x, features, left_child, right_child,
                                        cuda=self.__cuda)
        # FIXME hardcoded now
        after_conv_other = self.conv_module_list_other[0](trees_other)
        after_conv_hash_join = self.conv_module_list_hash_join[0](trees_hash_join)
        after_conv_nested_loop_join = self.conv_module_list_nested_loop_join[0](trees_nested_loop_join)
        after_conv_output = [after_conv_other, after_conv_hash_join, after_conv_nested_loop_join]
        after_conv_final = self.attention_merger(after_conv_output)
        out = self.out_mlp(after_conv_final)
        return out

    def cuda(self):
        self.__cuda = True
        return super().cuda()
