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
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )
        self.conv = nn.Sequential(
            TreeConv1d(32 + plan_size, 512),
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

    def forward(self, query_feats, trees, indexes):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors.  Shaped as
            [batch size, query dims].
          trees: The input plan features.  Shaped as
            [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        query_embs = self.query_mlp(query_feats.unsqueeze(1))

        query_embs = query_embs.transpose(1, 2)
       
        max_subtrees = trees.shape[-1]
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
     
        concat = torch.cat((query_embs, trees), axis=1)
      
        out = self.conv((concat, indexes))
       
        out = self.out_mlp(out)
        return out


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
def _make_preorder_ids_tree(curr,other_operators_set,hash_join_set,nested_loop_join_set, root_index=1):
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
    #assert other_operators_set  and hash_join_set  and nested_loop_join_set 

        
    if curr.node_type == 'Hash Join':
        hash_join_set.add(root_index)
    elif curr.node_type == 'Nested Loop':
        nested_loop_join_set.add(root_index)
    else:
        other_operators_set.add(root_index)
        
    if not curr.children:
        return (root_index, 0, 0), root_index
    lhs, lhs_max_id = _make_preorder_ids_tree(curr.children[0],  
                                              other_operators_set=other_operators_set,
                                              hash_join_set=hash_join_set,
                                              nested_loop_join_set=nested_loop_join_set,
                                              root_index=root_index + 1)
    rhs, rhs_max_id = _make_preorder_ids_tree(curr.children[1],        
                                              other_operators_set=other_operators_set,
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


# @profile
def _make_indexes(root):
    # Join(A, B) --> preorder_ids = (1, (2, 0, 0), (3, 0, 0))
    # Join(Join(A, B), C) --> preorder_ids = (1, (2, 3, 4), (5, 0, 0))
    other_operators_set = set()
    hash_join_set = set()
    nested_loop_join_set = set()
    preorder_ids, _ = _make_preorder_ids_tree(root,other_operators_set,hash_join_set,nested_loop_join_set)
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
    #now we have to divide the vecs into three vectors, which have the same shape as vecs
    # : vec_other_operators, vec_hash_join, vec_nested_loop_join
    #for exaple, if nested_loop_join_set is empty, hash_join_set contains "2", and other_operators_set contains "1" and "3"
    #and "4" and "5"
    node_ids = vecs[:, 0]  # extract all nodes ids
    vecs_other_operators = np.where(np.isin(node_ids, list(other_operators_set)), node_ids, 0).reshape(-1, 1)
    vecs_hash_join = np.where(np.isin(node_ids, list(hash_join_set)), node_ids, 0).reshape(-1, 1)
    vecs_nested_loop_join = np.where(np.isin(node_ids, list(nested_loop_join_set)), node_ids, 0).reshape(-1, 1)
    
    return vecs


# @profile
def _featurize_tree(curr_node, node_featurizer):

    def _bottom_up(curr):
    
    
        """Calls node_featurizer on each node exactly once, bottom-up."""
        if hasattr(curr, '__node_feature_vec'):
            return curr.__node_feature_vec
        if not curr.children:
            # this one will tackle with scan operator
            vec = node_featurizer.FeaturizeLeaf(curr)
            curr.__node_feature_vec = vec
            return vec
        left_vec = _bottom_up(curr.children[0])
        right_vec = _bottom_up(curr.children[1])
        # this one will tackle with join operator
        vec = node_featurizer.Merge(curr, left_vec, right_vec)
        curr.__node_feature_vec = vec
        return vec

    _bottom_up(curr_node)
    
    def append_vecs(node):
        # debug
        zero_vec = np.zeros(node.__node_feature_vec.size, dtype=np.float32)
        vec = getattr(node, '__node_feature_vec', zero_vec)
        vecs.append(vec)

        if node.node_type == "Hash Join":
            vecs_hash_join.append(vec)
            vecs_other_operators.append(zero_vec)
            vecs_nested_loop_join.append(zero_vec)
        elif node.node_type == "Nested Loop":
            vecs_nested_loop_join.append(vec)
            vecs_other_operators.append(zero_vec)
            vecs_hash_join.append(zero_vec)
        else:
            vecs_other_operators.append(vec)
            vecs_hash_join.append(zero_vec)
            vecs_nested_loop_join.append(zero_vec)
    
    vecs = []
    vecs_other_operators = []
    vecs_hash_join = []
    vecs_nested_loop_join = []
    # vecs = []
    # vecs_other_operators = []
    # vecs_hash_join = []
    # vecs_nested_loop_join = []
    # plans_lib.MapNode(curr_node,
    #                   lambda node: vecs.append(node.__node_feature_vec))
    # # Add a zero-vector at index 0.
    # ret = np.zeros((len(vecs) + 1, vecs[0].shape[0]), dtype=np.float32)
    # ret[1:] = vecs
        # 创建基础数组并填充第一个索引位置为全零向量
    plans_lib.MapNode(curr_node, append_vecs)
    num_nodes = len(vecs)
    vec_size = vecs[0].shape[0]
    ret = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)
    ret_other_operators = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)
    ret_hash_join = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)
    ret_nested_loop_join = np.zeros((num_nodes + 1, vec_size), dtype=np.float32)

    # 填充基础数组
    #import pprint
    ret[1:] = vecs
    #np.set_printoptions(threshold=np.inf)
    #print("ret:", pprint.pformat(ret))
    ret_other_operators[1:] = vecs_other_operators
    #print("ret_other_operators:", pprint.pformat(ret_other_operators))
    ret_hash_join[1:] = vecs_hash_join
    #print("ret_hash_join:", pprint.pformat(ret_hash_join))
    ret_nested_loop_join[1:] = vecs_nested_loop_join
    #print("ret_nested_loop_join:", pprint.pformat(ret_nested_loop_join))
    # if we want to check if the ret is the sum of the other three arrays
    # combined_array = ret_other_operators + ret_hash_join + ret_nested_loop_join
    # is_equal = np.allclose(ret[1:], combined_array[1:], atol=1e-7)
    # print("Is ret the sum of the other three arrays:", is_equal) #true
    
    # now we get the ret, ret_other_operators, ret_hash_join, ret_nested_loop_join
    return ret


# @profile
def make_and_featurize_trees(trees, node_featurizer):
    indexes = torch.from_numpy(_batch([_make_indexes(x) for x in trees])).long()
    trees = torch.from_numpy(
        _batch([_featurize_tree(x, node_featurizer) for x in trees
               ])).transpose(1, 2)
    return trees, indexes
