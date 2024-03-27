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
        self.conv_module_list_other = nn.ModuleList([self.create_conv_module(plan_size)])
        self.conv_module_list_hash_join = nn.ModuleList([self.create_conv_module(plan_size)])
        self.conv_module_list_nested_loop_join = nn.ModuleList([self.create_conv_module(plan_size)]) 
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
        
    def create_conv_module(self, input_size):
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
            new_module_other = self.create_conv_module(self.plan_size)
            new_module_other.to(query_feats.device)
            self.conv_module_list_other.append(new_module_other)
            idx_other = len(self.conv_module_list_other) - 1
        out_other = self.conv_module_list_other[idx_other]((concat,indexes_pos_feats))
        
        if idx_hash_join == -1:
            new_module_hash_join = self.create_conv_module(self.plan_size)
            new_module_hash_join.to(query_feats.device)
            self.conv_module_list_hash_join.append(new_module_hash_join)
            idx_hash_join = len(self.conv_module_list_hash_join) - 1
        out_hash_join = self.conv_module_list_hash_join[idx_hash_join]((concat_hash_join,hash_join_pos_feats))
        
        if idx_nested_loop_join == -1:
            new_module_nested_loop_join = self.create_conv_module(self.plan_size)
            new_module_nested_loop_join.to(query_feats.device)
            self.conv_module_list_nested_loop_join.append(new_module_nested_loop_join)
            idx_nested_loop_join = len(self.conv_module_list_nested_loop_join) - 1   
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
    return vecs

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
