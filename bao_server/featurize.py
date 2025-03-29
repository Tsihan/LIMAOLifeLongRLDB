import numpy as np

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES


class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return node["Node Type"] in JOIN_TYPES

def is_scan(node):
    return node["Node Type"] in LEAF_TYPES

class TreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return np.concatenate((arr, self.__stats(node)))
    
    def __featurize_scan(self, node):
        assert is_scan(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return (np.concatenate((arr, self.__stats(node))),
                self.__relation_name(node))


    def plan_to_feature_tree(self, plan):
        children = plan["Plans"] if "Plans" in plan else []

        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])

        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            return self.__featurize_scan(plan)

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))



def norm(x, lo, hi):
    return (np.log(x + 1) - lo) / (hi - lo)

def get_buffer_count_for_leaf(leaf, buffers):
    total = 0
    if "Relation Name" in leaf:
        total += buffers.get(leaf["Relation Name"], 0)

    if "Index Name" in leaf:
        total += buffers.get(leaf["Index Name"], 0)

    return total

class StatExtractor:
    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        res = []
        for f, lo, hi in zip(self.__fields, self.__mins, self.__maxs):
            if f not in inp:
                res.append(0)
            else:
                res.append(norm(inp[f], lo, hi))
        return res

def get_plan_stats(data):
    costs = []
    rows = []
    bufs = []
    
    def recurse(n, buffers=None):
        costs.append(n["Total Cost"])
        rows.append(n["Plan Rows"])
        if "Buffers" in n:
            bufs.append(n["Buffers"])

        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)

    for plan in data:
        recurse(plan["Plan"], buffers=plan.get("Buffers", None))

    costs = np.array(costs)
    rows = np.array(rows)
    bufs = np.array(bufs)
    
    costs = np.log(costs + 1)
    rows = np.log(rows + 1)
    bufs = np.log(bufs + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    bufs_min = np.min(bufs) if len(bufs) != 0 else 0
    bufs_max = np.max(bufs) if len(bufs) != 0 else 0

    if len(bufs) != 0:
        return StatExtractor(
            ["Buffers", "Total Cost", "Plan Rows"],
            [bufs_min, costs_min, rows_min],
            [bufs_max, costs_max, rows_max]
        )
    else:
        return StatExtractor(
            ["Total Cost", "Plan Rows"],
            [costs_min, rows_min],
            [costs_max, rows_max]
        )
        

def get_all_relations(data):
    all_rels = []
    
    def recurse(plan):
        if "Relation Name" in plan:
            yield plan["Relation Name"]

        if "Plans" in plan:
            for child in plan["Plans"]:
                yield from recurse(child)

    for plan in data:
        all_rels.extend(list(recurse(plan["Plan"])))
        
    return set(all_rels)

def get_featurized_trees(data):
    all_rels = get_all_relations(data)
    stats_extractor = get_plan_stats(data)

    t = TreeBuilder(stats_extractor, all_rels)
    trees = []

    for plan in data:
        tree = t.plan_to_feature_tree(plan)
        trees.append(tree)
            
    return trees

def _attach_buf_data(tree):
    if "Buffers" not in tree:
        return

    buffers = tree["Buffers"]

    def recurse(n):
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)
            return
        
        # it is a leaf
        n["Buffers"] = get_buffer_count_for_leaf(n, buffers)

    recurse(tree["Plan"])

class TreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        for t in trees:
            _attach_buf_data(t)
        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):
        for t in trees:
            _attach_buf_data(t)
        return [self.__tree_builder.plan_to_feature_tree(x["Plan"]) for x in trees]
    

    def num_operators(self):
        return len(ALL_TYPES)
    
    
    def transform_subtrees(self, trees):
        """
        对于每棵原始计划树（trees 中的每个元素），首先调用 plan_to_feature_tree 得到特征树，
        然后生成三份：
        - full_tree_list: 原始特征树（包装为单元素列表）；
        - nested_loop_list: 对特征树进行 masking，保留从第一个 Nested Loop 节点开始的信息，
                            在此之前的节点的特征向量全置零，但文本信息保持；
        - hash_join_list: 类似地，对 Hash Join 节点进行 masking。
        """
        # 预处理：先为每棵树调用 _attach_buf_data
        for t in trees:
            _attach_buf_data(t)
        # 通过 plan_to_feature_tree 得到每棵树的特征树（与 transform 一致）
        feature_trees = [self.__tree_builder.plan_to_feature_tree(x["Plan"]) for x in trees]

        # 定义内部的 masking 函数，递归处理节点
        def mask_tree(node, target_type, encountered=False):
            """
            对节点进行 masking：
            - 如果 node 是 join 节点（tuple 长度==3）：
                * 若尚未遇到目标且当前节点类型等于 target_type，则：
                    - 对左右子树递归调用 mask_tree，传入 encountered=True（此节点及其后代保留原始信息）
                * 若尚未遇到目标且当前节点不是目标，则：
                    - 将当前节点的特征向量替换为同形状全零数组，再对左右子树递归调用（保持 encountered=False）
                * 如果已遇到目标，则直接返回原节点
            - 如果 node 是叶节点（tuple 长度==2），则：  
                * 未遇到目标时，将特征向量置零；否则返回原节点
            - 其他情况，原样返回。
            """
            if not isinstance(node, tuple):
                return node
            # 尝试根据第一元素（假设为特征向量）判断节点类型
            node_type = "Unknown"
            if isinstance(node[0], np.ndarray):
                vec = node[0]
                if vec[ALL_TYPES.index("Nested Loop")] == 1:
                    node_type = "Nested Loop"
                elif vec[ALL_TYPES.index("Hash Join")] == 1:
                    node_type = "Hash Join"
                elif vec[ALL_TYPES.index("Merge Join")] == 1:
                    node_type = "Merge Join"
                elif vec[ALL_TYPES.index("Seq Scan")] == 1:
                    node_type = "Seq Scan"
                elif vec[ALL_TYPES.index("Index Scan")] == 1:
                    node_type = "Index Scan"
                elif vec[ALL_TYPES.index("Index Only Scan")] == 1:
                    node_type = "Index Only Scan"
                elif vec[ALL_TYPES.index("Bitmap Index Scan")] == 1:
                    node_type = "Bitmap Index Scan"
            if len(node) == 3:
                # join node
                if not encountered and node_type == target_type:
                    # 目标节点：保留本节点及其后代信息
                    return (node[0],
                            mask_tree(node[1], target_type, encountered=True),
                            mask_tree(node[2], target_type, encountered=True))
                elif encountered:
                    return node
                else:
                    # 未遇到目标，mask 当前节点特征向量（置零），继续递归
                    masked_vec = np.zeros_like(node[0])
                    left_masked = mask_tree(node[1], target_type, encountered=False)
                    right_masked = mask_tree(node[2], target_type, encountered=False)
                    return (masked_vec, left_masked, right_masked)
            elif len(node) == 2:
                # 叶节点
                if not encountered:
                    masked_vec = np.zeros_like(node[0])
                    return (masked_vec, node[1])
                else:
                    return node
            else:
                return node

        full_tree_list = []
        nested_loop_list = []
        hash_join_list = []
        for ft in feature_trees:
            # full_tree_list 保持原样，但包装为单元素列表
            full_tree_list.append(ft)
            # 对 Nested Loop 和 Hash Join 分别调用 mask_tree
            nested_loop_list.append(mask_tree(ft, "Nested Loop", encountered=False))
            hash_join_list.append(mask_tree(ft, "Hash Join", encountered=False))
            
        assert len(full_tree_list) == len(nested_loop_list) == len(hash_join_list)
        return full_tree_list, nested_loop_list, hash_join_list

