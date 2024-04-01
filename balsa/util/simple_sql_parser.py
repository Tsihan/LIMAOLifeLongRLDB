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

import re

import networkx as nx

def _CanonicalizeJoinCond(join_cond):
    """join_cond: 4-tuple"""
    t1, c1, t2, c2 = join_cond
    if t1 < t2:
        return join_cond
    return t2, c2, t1, c1


def _DedupJoinConds(join_conds):
    """join_conds: list of 4-tuple (t1, c1, t2, c2)."""
    canonical_join_conds = [_CanonicalizeJoinCond(jc) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def _GetJoinConds(sql):
    
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)  # 1st table
        \.     # the dot "."
        (\w+)  # 1st table column
        \s*    # optional whitespace
        [=!<>]+  # the comparison operator
        \s*    # optional whitespace
        (\w+)  # 2nd table
        \.     # the dot "."
        (\w+)  # 2nd table column
        """, re.VERBOSE)
    join_conds = join_cond_pat.findall(sql)
    return _DedupJoinConds(join_conds)

def _GetGraph(join_conds):
    g = nx.MultiGraph()
    for t1, c1, t2, c2 in join_conds:
        g.add_edge(t1, t2, join_keys={t1: c1, t2: c2})
        #print("now the length of graph's edges is: ", len(g.edges))
    return g


def _FormatJoinCond(tup):
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def ParseSql(sql, filepath=None, query_name=None):
    """Parses a SQL string into (nx.Graph, a list of join condition strings).

    Both use aliases to refer to tables.
    """
    # FIXME Qihan Zhang cannot parse correctly!
    join_conds = _GetJoinConds(sql)
    graph = _GetGraph(join_conds)
    join_conds = [_FormatJoinCond(c) for c in join_conds]
    return graph, join_conds


def simple_encode_sql(sql_str):
#     sql_str = """
# SELECT
#     l.l_shipmode,
#     SUM(CASE
#         WHEN o.o_orderpriority = '1-URGENT'
#             OR o.o_orderpriority = '3-MEDIUM'
#             THEN 1
#         ELSE 0
#     END) AS high_line_count,
#     SUM(CASE
#         WHEN o.o_orderpriority <> '1-URGENT'
#             AND o.o_orderpriority <> '3-MEDIUM'
#             THEN 1
#         ELSE 0
#     END) AS low_line_count
# FROM
#     orders AS o,
#     lineitem AS l
# WHERE
#     o.o_orderkey = l.l_orderkey
#     AND l.l_shipmode IN ('RAIL', 'AIR')
#     AND l.l_commitdate < l.l_receiptdate
#     AND l.l_shipdate < l.l_commitdate
#     AND l.l_receiptdate >= DATE '1993-01-01'
#     AND l.l_receiptdate < DATE '1993-01-01' + INTERVAL '3' YEAR
# GROUP BY
#     l.l_shipmode
# ORDER BY
#     l.l_shipmode;
# """

    # One-hot vector [GROUP BY, ORDER BY, Aggregate Function, Subquery]
    features = [0, 0, 0, 0]

    # Check for GROUP BY
    if re.search(r"\bGROUP BY\b", sql_str, re.IGNORECASE):
        features[0] = 1

    # Check for ORDER BY
    if re.search(r"\bORDER BY\b", sql_str, re.IGNORECASE):
        features[1] = 1

    # Check for aggregate functions
    aggregate_functions = [
    "SUM", "COUNT", "AVG", "MIN", "MAX"
    ]
    for func in aggregate_functions:
        if re.search(r"\b" + func + r"\b", sql_str, re.IGNORECASE):
            features[2] = 1
            break

    # Check for subqueries
    subquery_patterns = [
    r"\bSELECT\b.*\bFROM\b.*\bSELECT\b",  # Nested SELECT
    r"\bEXISTS\b", r"\bIN\b", r"\bANY\b", r"\bSOME\b", r"\bALL\b"  # Keywords
    ]
    for pattern in subquery_patterns:
        if re.search(pattern, sql_str, re.IGNORECASE | re.DOTALL):
            features[3] = 1
            break

    print("Features: [GROUP BY, ORDER BY, Aggregate Function, Subquery]")
    print("Vector: ", features)
    return features
    


