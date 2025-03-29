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
    
    def transform_subtrees(self, trees):
        """
        对于每棵特征树（trees 中的每个元素），返回三个部分：
        - full_tree_list: 从原树的根节点开始，向下保留上层信息，
            遇到 Nested Loop 或 Hash Join 时截断（即不展开其子树，但保留该节点），
            这样保证了上层信息不丢失。
        - nested_loop_list: 该树中所有最大化的 Nested Loop 子树（遇到一个 Nested Loop 后，不继续收集其内部 Nested Loop 节点）。
        - hash_join_list: 同理，收集最大化的 Hash Join 子树。
        """
        full_tree_list = []
        nested_loop_list = []
        hash_join_list = []

        # 辅助函数：根据特征向量判断当前节点的类型
        def get_node_type(node):
            if isinstance(node, tuple) and len(node) == 3:
                vec = node[0]
                if vec[ALL_TYPES.index("Nested Loop")] == 1:
                    return "Nested Loop"
                elif vec[ALL_TYPES.index("Hash Join")] == 1:
                    return "Hash Join"
                elif vec[ALL_TYPES.index("Merge Join")] == 1:
                    return "Merge Join"
                elif vec[ALL_TYPES.index("Seq Scan")] == 1:
                    return "Seq Scan"
                elif vec[ALL_TYPES.index("Index Scan")] == 1:
                    return "Index Scan"
                elif vec[ALL_TYPES.index("Index Only Scan")] == 1:
                    return "Index Only Scan"
                elif vec[ALL_TYPES.index("Bitmap Index Scan")] == 1:
                    return "Bitmap Index Scan"
            # 叶子节点视为 "Scan"
            return "Unknown"

        # 构造截断后的全树：从根开始递归
        # 当遇到 Nested Loop 或 Hash Join 节点时，不展开其子树（用 None 代替），但仍保留该节点
        def build_full_tree(node):
            node_type = get_node_type(node)
            # 如果是叶子，或者是 Nested Loop / Hash Join 节点，则截断（注意：若 node 为 join 节点，返回时将子树置为空）
            if node_type in ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan", "Nested Loop", "Hash Join"]:
                if isinstance(node, tuple) and len(node) == 3:
                    return (node[0])
                else:
                    return node
            # 如果是 Merge Join，则继续递归构造左右子树
            if isinstance(node, tuple) and len(node) == 3:
                left_subtree = build_full_tree(node[1])
                right_subtree = build_full_tree(node[2])
                return (node[0], left_subtree, right_subtree)
            return node

        # 递归收集最大化目标类型的子树
        # 一旦遇到目标类型的节点，则将其加入列表，并不再向下遍历该分支
        def collect_maximal_join_subtrees(node, target_type, collected):
            if isinstance(node, tuple) and len(node) == 3:
                node_type = get_node_type(node)
                if node_type == target_type:
                    collected.append(node)
                    return  # 不再继续深入该分支
                else:
                    collect_maximal_join_subtrees(node[1], target_type, collected)
                    collect_maximal_join_subtrees(node[2], target_type, collected)
            # 叶子节点无需处理

        for ft in trees:
            # full_tree_list 中保留截断后的上层信息（始终以原树根节点为根），包装成单元素列表
            truncated_tree = build_full_tree(ft)
            full_tree_list.append([truncated_tree])

            # nested_loop_list：收集最大化的 Nested Loop 子树
            nl_subtrees = []
            collect_maximal_join_subtrees(ft, "Nested Loop", nl_subtrees)
            nested_loop_list.append(nl_subtrees)

            # hash_join_list：收集最大化的 Hash Join 子树
            hj_subtrees = []
            collect_maximal_join_subtrees(ft, "Hash Join", hj_subtrees)
            hash_join_list.append(hj_subtrees)
            
        for i in range(len(full_tree_list)):
            print(f"length of full_tree_list[{i}]:", len(full_tree_list[i]))
            print(f"length of nested_loop_list[{i}]:", len(nested_loop_list[i]))
            print(f"length of hash_join_list[{i}]:", len(hash_join_list[i]))

        assert len(full_tree_list) == len(nested_loop_list) == len(hash_join_list)
        return full_tree_list, nested_loop_list, hash_join_list

    def num_operators(self):
        return len(ALL_TYPES)
