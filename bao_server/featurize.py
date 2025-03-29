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
        - full_tree_list: 从原树的根节点开始向下遍历，直到遇到 Nested Loop 或 Hash Join 节点，
                            遇到后，对该节点及其所有子孙节点的 feature vector 用全零数组替换，
                            文本信息保留。
        - nested_loop_list: 采用原有策略，对 Nested Loop 节点之后的子树保留信息，其上层 mask 掉。
        - hash_join_list: 类似地，对 Hash Join 节点进行 masking。
        """
        # 预处理：对每棵树先调用 _attach_buf_data
        for t in trees:
            _attach_buf_data(t)
        # 调用 plan_to_feature_tree 得到每棵树的特征树（与 transform 输出一致）
        feature_trees = [self.__tree_builder.plan_to_feature_tree(x["Plan"]) for x in trees]

        # 定义一个辅助函数 mask_all，用于对整个子树进行全零 mask（但保留文本信息）
        def mask_all(node):
            if not isinstance(node, tuple):
                return node
            if len(node) == 3:
                masked_vec = np.zeros_like(node[0])
                return (masked_vec,
                        mask_all(node[1]),
                        mask_all(node[2]))
            elif len(node) == 2:
                masked_vec = np.zeros_like(node[0])
                return (masked_vec, node[1])
            else:
                return node

        # 新的 full_tree_list 生成函数：
        # 从根节点开始向下遍历，一旦遇到 Nested Loop 或 Hash Join 节点，就停止继续向下，
        # 并将该节点及其所有子孙的 feature vector 全部置零（但文本信息保留）
        def mask_full_tree(node):
            if not isinstance(node, tuple):
                return node
            # 判断节点类型
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
            # 如果当前节点是 Nested Loop 或 Hash Join，则返回全零 mask 整个子树
            if node_type in ["Nested Loop", "Hash Join"]:
                return mask_all(node)
            # 否则，如果是 join 节点，递归处理左右子树
            if len(node) == 3:
                return (node[0], mask_full_tree(node[1]), mask_full_tree(node[2]))
            elif len(node) == 2:
                # 叶节点直接返回
                return node
            else:
                return node

        # 生成 full_tree_list：对每棵特征树调用 mask_full_tree
        full_tree_list = [mask_full_tree(ft) for ft in feature_trees]

        # 旧的 mask_tree，用于生成 nested_loop_list 与 hash_join_list
        def mask_tree(node, target_type, encountered=False):
            if not isinstance(node, tuple):
                return node
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
                if not encountered and node_type == target_type:
                    return (node[0],
                            mask_tree(node[1], target_type, encountered=True),
                            mask_tree(node[2], target_type, encountered=True))
                elif encountered:
                    return node
                else:
                    masked_vec = np.zeros_like(node[0])
                    left_masked = mask_tree(node[1], target_type, encountered=False)
                    right_masked = mask_tree(node[2], target_type, encountered=False)
                    return (masked_vec, left_masked, right_masked)
            elif len(node) == 2:
                if not encountered:
                    masked_vec = np.zeros_like(node[0])
                    return (masked_vec, node[1])
                else:
                    return node
            else:
                return node

        nested_loop_list = [mask_tree(ft, "Nested Loop", encountered=False) for ft in feature_trees]
        hash_join_list = [mask_tree(ft, "Hash Join", encountered=False) for ft in feature_trees]

        # 返回三个列表，每个列表中对应一棵树（full_tree_list 此时是经过新 mask_full_tree 处理后的结果）
        assert len(full_tree_list) == len(nested_loop_list) == len(hash_join_list)
        return full_tree_list, nested_loop_list, hash_join_list
