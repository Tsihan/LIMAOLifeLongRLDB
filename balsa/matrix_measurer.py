import numpy as np

def calculate_difference(matrix1, matrix2):
    diff = abs(len(matrix1) - len(matrix2))
    length = min(len(matrix1), len(matrix2))
    return sum([1 for i in range(length) if matrix1[i] != matrix2[i]]) + diff

def get_final_value(difference_operators, difference_indexes, difference_sql_feature, l2_distance):
    return 1 * difference_operators + 1 * difference_indexes + 50 * difference_sql_feature + 5 * l2_distance

def compute_difference(operators_env_matrix_query, indexes_env_matrix_query, query_enc_matrix_query, sql_feature_encode_matrix_query,
                       operators_env_matrix_feature, indexes_env_matrix_feature, query_enc_matrix_feature, sql_feature_encode_matrix_feature):
    difference_operators = calculate_difference(operators_env_matrix_query, operators_env_matrix_feature)
    difference_indexes = calculate_difference(indexes_env_matrix_query, indexes_env_matrix_feature)
    difference_sql_feature = calculate_difference(sql_feature_encode_matrix_query, sql_feature_encode_matrix_feature)
    min_length = min(len(query_enc_matrix_query), len(query_enc_matrix_feature))
    query_enc_matrix_query = query_enc_matrix_query[:min_length]
    query_enc_matrix_feature = query_enc_matrix_feature[:min_length]
    l2_distance = np.linalg.norm(np.array(query_enc_matrix_query) - np.array(query_enc_matrix_feature))
    return get_final_value(difference_operators, difference_indexes, difference_sql_feature, l2_distance)
       
  
 
