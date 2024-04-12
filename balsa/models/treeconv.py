# Copyright 2022 The Balsa Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn

from balsa.util import plans_lib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# JOB mean representative
operators_env_matrix1 =  np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0])
indexes_env_matrix1 = np.array([1, 2, 16, 2, 3, 15, 3, 4, 13, 4, 4, 11, 5, 4, 10, 6, 5, 9, 7, 4, 7, 8, 4, 7, 2, 0, 0,
 7, 3, 5, 8, 3, 4, 9, 3, 3, 9, 2, 3, 10, 1, 2, 9, 1, 2, 10, 1, 2, 7, 1, 1, 0, 0, 0, 8,
 0, 0, 6, 0, 0, 6, 0, 0, 5, 0, 0, 6, 0, 0, 4, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0,
 2, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
query_enc_matrix1 = np.array([0.010638297872340425, 0.1947722525638298, 0.010638297872340425, 0.0, 0.010638297872340425,
 0.24159679912553186, 0.11830316155625424, 0.05851063829787234, 0.05053191489361702, 0.20455046004985114,
 0.009675496808510639, 0.031914893617021274, 0.17819148936170212, 0.19148936170212766,
 0.033797778430638296, 0.002636038617021276, 0.002824327127659573, 0.00028243268617021277,
 0.03193352669434299, 0.04255319360638298, 0.006079027659574468, 0.006079027659574468,
 0.05496453914893615, 0.43134618426226584, 0.031914893617021274, 0.031914893617021274,
 0.07379678908415958, 0.2332790913829787, 0.031914893617021274, 0.01566663936170213, 0.0425531914893617,
 0.6382978723404256, 0.19148936170212766, 0.1331173170896915, 0.02127659574468085, 0.03221491755711457,
 0.02393617088936169, 0.47566778605380095, 0.05319148936170213, 0.027951585744680853])
sql_feature_encode_matrix1 = np.array([0, 0, 1, 0])
 
# IMDB Bao mean representative
operators_env_matrix2 =  np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 3, 2, 3, 3, 3, 2, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1])
indexes_env_matrix2 = np.array([1, 2, 18, 2, 3, 16, 3, 4, 17, 4, 4, 13, 5, 5, 13, 6, 6, 16, 7, 7, 15, 8, 7, 13,
 0, 0, 0, 9, 7, 12, 10, 8, 11, 10, 5, 8, 11, 3, 5, 12, 4, 5, 11, 3, 4, 12, 2, 3,
 13, 2, 2, 0, 0, 0, 14, 0, 0, 12, 0, 0, 13, 0, 0, 13, 0, 0, 13, 0, 0, 8, 0, 0, 8,
 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0])
query_enc_matrix2 = np.array([0.27, 0.6796959436829999, 0.060801977140000005, 0.005, 0.005, 0.037294178105212, 0.065, 0.02,
 0.0010619471000000002, 0.14504424822469983, 0.003982301449699998, 0.0019469026499999993,
 0.0019469026499999993, 0.0016814159249999995, 0.40001818588328175, 0.22142857774999986, 0.03, 0.31,
 0.11166872926999999, 0.007517643310420002, 0.0014119037826399998, 0.07, 0.0001893864962,
 0.0001834880937999999, 0.06, 0.76, 0.03, 0.22536291416328005, 0.009822547198, 0.10916666906000001,
 0.6133784937580998, 0.03, 0.03])
sql_feature_encode_matrix2 = np.array([0, 0, 1, 0])

# JOB changed mean representative
operators_env_matrix3 =  np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
indexes_env_matrix3 = np.array([1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 11, 5, 4, 10, 6, 5, 9, 7, 4, 6, 8, 4, 6, 2, 0, 0, 7,
 3, 4, 8, 2, 3, 9, 2, 3, 9, 2, 2, 10, 1, 2, 9, 1, 1, 10, 1, 1, 7, 0, 0, 0, 0, 0, 8, 0, 0,
 5, 0, 0, 6, 0, 0, 4, 0, 0, 5, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
query_enc_matrix3 = np.array([0.010526315789473684, 0.17166938674736842, 0.021052631578947368, 0.010526315789473684,
 0.031578947368421054, 0.2388503383557894, 0.12800837582379948, 0.05, 0.039473684210526314,
 0.19318683248483579, 0.003948404210526316, 0.021052631578947368, 0.2, 0.16842105263157894,
 0.03344201234189474, 0.0027014440789473676, 0.002701444105263157, 0.00018630647368421055,
 0.0631753902157956, 0.039097746305263165, 0.003007518947368421, 0.003007518947368421,
 0.05204678372631577, 0.4522579790700315, 0.021052631578947368, 0.021052631578947368,
 0.08108109977885265, 0.22141186853052625, 0.021052631578947368, 0.010278366315789473,
 0.042105263157894736, 0.6210526315789474, 0.16842105263157894, 0.1525861300823684,
 0.021265262315789474, 0.021349497372302843, 0.01491228151157895, 0.5011503272083473,
 0.042105263157894736, 0.02336705694736842])
sql_feature_encode_matrix3 = np.array([1, 1, 1, 0])   

# IMDB Bao changed mean representative

operators_env_matrix4 =  np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1])
indexes_env_matrix4 = np.array([1, 2, 18, 2, 3, 16, 3, 4, 17, 4, 4, 13, 5, 5, 13, 6, 6, 16, 7, 7, 15, 8, 7, 13,
 0, 0, 0, 9, 7, 12, 9, 8, 11, 10, 6, 8, 11, 3, 5, 12, 4, 5, 11, 3, 4, 12, 2, 3,
 13, 2, 2, 0, 0, 0, 14, 0, 0, 12, 0, 0, 13, 0, 0, 13, 0, 0, 14, 0, 0, 8, 0, 0, 8,
 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4,
 0, 0])
query_enc_matrix4 = np.array([0.2755102040816326, 0.693567289472449, 0.06204283381632653, 0.00510204081632653,
 0.00510204081632653, 0.03805528378082858, 0.0663265306122449, 0.02040816326530612,
 0.0010836194897959185, 0.12759617165785697, 0.004063572907857141, 0.001806032142857142,
 0.001806032142857142, 0.0017157305357142852, 0.38777365906457323, 0.22011662423469375,
 0.030612244897959183, 0.3163265306122449, 0.11394768292857141, 0.007042247194306125,
 0.0014407181455510204, 0.07142857142857142, 0.00017568320612244896, 0.0001702115897959183,
 0.061224489795918366, 0.7551020408163265, 0.030612244897959183, 0.22996215730946945,
 0.010023007344897958, 0.11139456026530614, 0.614564113732755, 0.030612244897959183,
 0.030612244897959183])
sql_feature_encode_matrix4 = np.array([1, 1, 1, 0])  

class TreeConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.

    Value is either cost or latency.
    """
    
    def __init__(self, feature_size, plan_size, label_size, version=None):
        super(TreeConvolution, self).__init__()
        # None: default
        assert version is None, version
        self.attention_merger_3 = AttentionMerger(input_dim=128,input_num=3)
        # record it for later use
        self.plan_size = plan_size
        
        # Initialize dictionaries to store features for each module
        self.other_features = {}
        self.hash_join_features = {}
        self.nested_loop_join_features = {}

        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        
                # 初始化三个模块列表
        self.conv_module_list_other = nn.ModuleList([self.create_conv_module(plan_size,1,0,'initial')])
        self.conv_module_list_hash_join = nn.ModuleList([self.create_conv_module(plan_size,2,0,'initial')])
        self.conv_module_list_nested_loop_join = nn.ModuleList([self.create_conv_module(plan_size,3,0,'initial')]) 
        self.out_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )
        self.reset_weights()
    # create a new submodule and initialize its four feature matrices
    def create_conv_module(self, input_size, module_type,current_new_index = 0,caller = 'initial'):
        if module_type == 1:
            if caller == 'initial':
                self.other_features[current_new_index] = (operators_env_matrix1, indexes_env_matrix1, query_enc_matrix1, sql_feature_encode_matrix1)
            else:
                self.other_features[current_new_index] = (operators_env_matrix2, indexes_env_matrix2, query_enc_matrix2, sql_feature_encode_matrix2)
        elif module_type == 2:
            if caller == 'initial':
                self.hash_join_features[current_new_index] = (operators_env_matrix1, indexes_env_matrix1, query_enc_matrix1, sql_feature_encode_matrix1)
            else:
                self.hash_join_features[current_new_index] = (operators_env_matrix2, indexes_env_matrix2, query_enc_matrix2, sql_feature_encode_matrix2)
        else:
            if caller == 'initial':
                self.nested_loop_join_features[current_new_index] = (operators_env_matrix1, indexes_env_matrix1, query_enc_matrix1, sql_feature_encode_matrix1)
            else:
                self.nested_loop_join_features[current_new_index] = (operators_env_matrix2, indexes_env_matrix2, query_enc_matrix2, sql_feature_encode_matrix2)
        return nn.Sequential(
            TreeConv1d(32 + input_size, 512),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 256),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(256, 128),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),
            )
    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)

    
    def forward(self, idx_other,idx_hash_join,idx_nested_loop_join,
                query_feats,trees_feats,hash_join_feats,nested_loop_join_feats,indexes_pos_feats,
                                          hash_join_pos_feats,nested_loop_join_pos_feats
                                          ):
        """Forward pass.

        Args:


        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
           
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
            
            
        max_subtrees = trees_feats.shape[-1]
        max_subtrees_hash_join = hash_join_feats.shape[-1]
        max_subtrees_nested_loop_join = nested_loop_join_feats.shape[-1]
            
            
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1], max_subtrees)
        query_embs_hash_join = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                        max_subtrees_hash_join)
        query_embs_nested_loop_join = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                        max_subtrees_nested_loop_join)
            
            
        concat = torch.cat((query_embs, trees_feats), axis=1)
        concat_hash_join = torch.cat((query_embs_hash_join, hash_join_feats), axis=1)
        concat_nested_loop_join = torch.cat((query_embs_nested_loop_join, nested_loop_join_feats), axis=1)
        
        if idx_other == -1:
            new_module_other = self.create_conv_module(self.plan_size,1,len(self.conv_module_list_other),'not')
            new_module_other.to(query_feats.device)
            self.conv_module_list_other.append(new_module_other)
            idx_other = len(self.conv_module_list_other) - 1
            print("created a new module and initialize the weights for other")
        out_other = self.conv_module_list_other[idx_other]((concat,indexes_pos_feats))
        
        if idx_hash_join == -1:
            new_module_hash_join = self.create_conv_module(self.plan_size,2,len(self.conv_module_list_hash_join),'not')
            new_module_hash_join.to(query_feats.device)
            self.conv_module_list_hash_join.append(new_module_hash_join)
            idx_hash_join = len(self.conv_module_list_hash_join) - 1
            print("created a new module and initialize the weights for hash join")
        out_hash_join = self.conv_module_list_hash_join[idx_hash_join]((concat_hash_join,hash_join_pos_feats))
        
        if idx_nested_loop_join == -1:
            new_module_nested_loop_join = self.create_conv_module(self.plan_size,3,len(self.conv_module_list_nested_loop_join),'not')
            new_module_nested_loop_join.to(query_feats.device)
            self.conv_module_list_nested_loop_join.append(new_module_nested_loop_join)
            idx_nested_loop_join = len(self.conv_module_list_nested_loop_join) - 1 
            print("created a new module and initialize the weights for nested loop join")  
        out_nested_loop_join = self.conv_module_list_nested_loop_join[idx_nested_loop_join]((concat_nested_loop_join,nested_loop_join_pos_feats))
        
        conv_outputs = [out_other, out_hash_join, out_nested_loop_join]
        out_combined = self.attention_merger_3(conv_outputs)
        out = self.out_mlp(out_combined)
        return out



class AttentionMerger(nn.Module):
    def __init__(self, input_dim,input_num):
        super(AttentionMerger, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_num, 1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, conv_outputs):
    # conv_outputs: list of tensors from the 3 conv layers, each [batch size, feature_dim]
        stacked_outputs = torch.stack(conv_outputs, dim=0)  # Shape: [3, batch size, feature_dim]
        weights = self.softmax(self.attention_weights)  # Shape: [3, 1]
    
    # 不需要为weights增加两个维度，而是直接应用weights并保持输出维度不变
    # 使用broadcasting机制对stacked_outputs的每个feature_dim进行加权
    # 首先调整weights的形状以匹配stacked_outputs的broadcasting要求
        weights = weights.view(-1, 1, 1)  # Adjusted shape for broadcasting: [3, 1, 1]
        weighted_outputs = weights * stacked_outputs  # Broadcasting weights to each feature_dim
    
    # 加权求和，不增加额外的维度，保持[batch size, feature_dim]形状
        weighted_sum = torch.sum(weighted_outputs, dim=0)  # Shape: [batch size, feature_dim]
        return weighted_sum
   
class TreeConv1d(nn.Module):
    """Conv1d adapted to tree data."""

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=3, stride=3)

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = self.weights(
            torch.gather(data, 2,
                         indexes.expand(-1, -1, self._in_dims).transpose(1, 2)))
        zeros = torch.zeros((data.shape[0], self._out_dims),
                            device=DEVICE).unsqueeze(2)
        feats = torch.cat((zeros, feats), dim=2)
        return feats, indexes


class TreeMaxPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].max(dim=2).values


class TreeAct(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return self.activation(trees[0]), trees[1]


class TreeStandardize(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        mu = torch.mean(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        s = torch.std(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        standardized = (trees[0] - mu) / (s + 1e-5)
        return standardized, trees[1]


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


# @profile
def _batch(data):
    lens = [vec.shape[0] for vec in data]
    if len(set(lens)) == 1:
        # Common path.
        return np.asarray(data)
    xs = np.zeros((len(data), np.max(lens), data[0].shape[1]), dtype=np.float32)
    for i, vec in enumerate(data):
        xs[i, :vec.shape[0], :] = vec
    return xs


# @profile

def get_last_preorder_index(curr, root_index=1):
    """Returns the index of the last node in the subtree rooted at the current node
    based on preorder traversal.

    Args:
        curr: The current node in the tree.
        root_index: The preorder index of the current node.

    Returns:
        The index of the last node in the preorder traversal of the subtree rooted at curr.
    """
    if not curr.children:
        # If the current node is a leaf, it is the last node in its subtree.
        return root_index

    # Recursively find the last index in the left and right subtrees.
    last_index_left = get_last_preorder_index(curr.children[0], root_index + 1)
    last_index_right = get_last_preorder_index(curr.children[1], last_index_left + 1)

    # The last index of the current subtree is the last index of the right subtree.
    return last_index_right

def _make_preorder_ids_tree(curr,hash_join_set,nested_loop_join_set, root_index=1):
    """Returns a tuple containing a tree of preorder positional IDs.

    Returns (tree structure, largest id under me).  The tree structure itself
    (the first slot) is a 3-tuple:

    If curr is a leaf:
      tree structure is (my id, 0, 0) (note that valid IDs start with 1)
    Else:
      tree structure is
        (my id, tree structure for LHS, tree structure for RHS).

    This function traverses each node exactly once (i.e., O(n) time complexity).
    
    """
 
    last_index = get_last_preorder_index(curr, root_index)
        
    if curr.node_type == 'Hash Join':
        for i in range(root_index,last_index+1):
            hash_join_set.add(i)
        
    elif curr.node_type == 'Nested Loop':
        for i in range(root_index,last_index+1):
            nested_loop_join_set.add(i)
        

        
    if not curr.children:
        return (root_index, 0, 0), root_index
    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0],   
                                              hash_join_set=hash_join_set,
                                              nested_loop_join_set=nested_loop_join_set,
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1],           
                                              hash_join_set=hash_join_set,
                                              nested_loop_join_set=nested_loop_join_set,
                                               root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id


# @profile
def _walk(curr, vecs):
    if curr[1] == 0:
        # curr is a leaf.
        vecs.append(curr)
    else:
        vecs.append((curr[0], curr[1][0], curr[2][0]))
        _walk(curr[1], vecs)
        _walk(curr[2], vecs)

def _make_preorder_ids_tree_environment(curr, root_index=1):


    if not curr.children:
        return (root_index, 0, 0), root_index
    lhs, lhs_max_id = _make_preorder_ids_tree_environment(curr.children[0],
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree_environment(curr.children[1],
                                              root_index=lhs_max_id + 1)
    return (root_index, lhs, rhs), rhs_max_id

def _make_indexes_environment(root):
    # Join(A, B) --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C) --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    preorder_ids, _ = _make_preorder_ids_tree_environment(root)
    vecs = []
    _walk(preorder_ids, vecs)
    # Continuing with the Join(A,B) example:
    # Preorder traversal _walk() produces
    #   [1, 2, 3]
    #   [2, 0, 0]
    #   [3, 0, 0]
    # which would be reshaped into
    #   array([[1],
    #          [2],
    #          [3],
    #          [2],
    #          [0],
    #          [0],
    #    ...,
    #          [0]])
    vecs = np.asarray(vecs).reshape(-1, 1)
    return vecs.flatten()

def _make_indexes(root):
    hash_join_set = set()
    nested_loop_join_set = set()
    preorder_ids, _ = _make_preorder_ids_tree(root, hash_join_set, nested_loop_join_set)
    vecs = []
    _walk(preorder_ids, vecs)
    vecs = np.asarray(vecs).reshape(-1, 1)

    node_ids = vecs[:, 0]  # extract all nodes ids

    # Create masks for hash join and nested loop nodes
    hash_join_mask = np.isin(node_ids, list(hash_join_set))
    nested_loop_join_mask = np.isin(node_ids, list(nested_loop_join_set))

    # Update vecs_hash_join and vecs_nested_loop_join to include only relevant nodes
    vecs_hash_join = np.where(hash_join_mask, node_ids, 0).reshape(-1, 1)
    vecs_nested_loop_join = np.where(nested_loop_join_mask, node_ids, 0).reshape(-1, 1)

    # Update vecs to include only nodes that are not part of hash join or nested loop subtrees
    vecs = np.where(~hash_join_mask & ~nested_loop_join_mask, node_ids, 0).reshape(-1, 1)

    return vecs, vecs_hash_join, vecs_nested_loop_join
# @profile


def _featurize_tree(curr_node, node_featurizer):
    def _bottom_up(curr):
        """Calls node_featurizer on each node exactly once, bottom-up."""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            vec = node_featurizer.FeaturizeLeaf(curr)
            curr.__node_feature_vec = vec
            return vec
        left_vec = _bottom_up(curr.children[0])
        right_vec = _bottom_up(curr.children[1])
        vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)
    
    def add_subtree_vecs(node, vecs_list):
        if hasattr(node, '__node_feature_vec'):
            vecs_list.append(node.__node_feature_vec)
        for child in node.children:
            add_subtree_vecs(child, vecs_list)
             
    def append_vecs(node):
        zero_vec = np.zeros(node.__node_feature_vec.size, dtype=np.float32)
        vec = getattr(node, '__node_feature_vec', zero_vec)
        
        if node in hash_join_subtree_nodes:
            vecs_hash_join.append(vec)
        else:
            vecs_hash_join.append(zero_vec)

        if node in nested_loop_subtree_nodes:
            vecs_nested_loop_join.append(vec)
        else:
            vecs_nested_loop_join.append(zero_vec)

        if node not in hash_join_subtree_nodes and node not in nested_loop_subtree_nodes:
            vecs.append(vec)
        else:
            vecs.append(zero_vec)
            
    def add_subtree_nodes(node, nodes_set):
        """Recursively adds all nodes in the subtree rooted at the given node to the specified set."""
        nodes_set.add(node)
        for child in node.children:
            add_subtree_nodes(child, nodes_set)

    def mark_subtree_nodes(node):
        """Mark all nodes in the subtree rooted at a Hash Join or Nested Loop node."""
        if node.node_type == "Hash Join":
            add_subtree_nodes(node, hash_join_subtree_nodes)
        elif node.node_type == "Nested Loop":
            add_subtree_nodes(node, nested_loop_subtree_nodes)
        for child in node.children:
            mark_subtree_nodes(child)
      
      
      
      
    
    
    hash_join_subtree_nodes = set()
    nested_loop_subtree_nodes = set()        
    mark_subtree_nodes(curr_node)        
    vecs = []
    vecs_hash_join = []
    vecs_nested_loop_join = []
   

    plans_lib.MapNode(curr_node, append_vecs)

    num_nodes = len(vecs)
    vec_size = vecs[0].shape[0]
    ret = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)
    ret_hash_join = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)
    ret_nested_loop_join = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)

    ret[1:] = vecs
    ret_hash_join[1:len(vecs_hash_join) + 1] = vecs_hash_join
    ret_nested_loop_join[1:len(vecs_nested_loop_join) + 1] = vecs_nested_loop_join

    return ret, ret_hash_join, ret_nested_loop_join


def make_and_featurize_trees(trees, node_featurizer):
    # 初始化列表以存储来自_make_indexes和_featurize_tree的不同向量
    indexes_list = []
   
    hash_join_indexes_list = []
    nested_loop_join_indexes_list = []

    trees_list = []
    
    hash_join_trees_list = []
    nested_loop_join_trees_list = []

    # 循环处理每棵树
    for tree in trees:
        # 调用_make_indexes函数并收集输出
        index_components = _make_indexes(tree)
        indexes_list.append(index_components[0])
        hash_join_indexes_list.append(index_components[1])
        nested_loop_join_indexes_list.append(index_components[2])

        # 调用_featurize_tree函数并收集输出
        tree_components = _featurize_tree(tree, node_featurizer)
        trees_list.append(tree_components[0])
     
        hash_join_trees_list.append(tree_components[1])
        nested_loop_join_trees_list.append(tree_components[2])

    # 批量化索引数据并转换为长整数张量
    indexes_batch = torch.from_numpy(_batch(indexes_list)).long()
   
    hash_join_indexes_batch = torch.from_numpy(_batch(hash_join_indexes_list)).long()
    nested_loop_join_indexes_batch = torch.from_numpy(_batch(nested_loop_join_indexes_list)).long()

    # 批量化特征数据并转换为浮点数张量
    trees_batch = torch.from_numpy(_batch(trees_list)).transpose(1, 2).float()
   
    hash_join_trees_batch = torch.from_numpy(_batch(hash_join_trees_list)).transpose(1, 2).float()
    nested_loop_join_trees_batch = torch.from_numpy(_batch(nested_loop_join_trees_list)).transpose(1, 2).float()

    # 返回所有的批量树和索引
    return (
        trees_batch,
        hash_join_trees_batch,
        nested_loop_join_trees_batch,
        indexes_batch,
        hash_join_indexes_batch,
        nested_loop_join_indexes_batch
    )
