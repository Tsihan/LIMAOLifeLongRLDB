import numpy as np
# may be set a threshold to 100.
# operators_env_matrix1 =  [1 ,1 ,1, 1 ,1, 1 ,3 ,4, 4, 4, 4, 5 ,3]
# indexes_env_matrix1 = [1, 2, 13, 2, 3, 12, 3, 4, 11, 4, 5, 10, 5, 6, 9, 6, 7, 8, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0]
# query_enc_matrix1= [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 2.7590511e-08, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.6103013e-01,
# 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 1.0000000e+00, 6.9441271e-01, 0.0000000e+00, 0.0000000e+00]
# sql_feature_encode_matrix1= [0, 0, 1, 0]

# operators_env_matrix2=  [1, 1 ,1 ,1 ,1 ,3 ,1, 1 ,3, 4, 4 ,4 ,4, 4 ,3]
# indexes_env_matrix2 = [1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 12, 5, 6, 7, 6, 0, 0, 7, 8, 11, 8, 9, 10, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0]
# query_enc_matrix2 = [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.4470397e-01,
# 0.0000000e+00, 0.0000000e+00, 2.5000000e-01, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.4532309e-06, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 5.5555556e-02, 4.8416656e-01, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 1.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 3.6069322e-01, 0.0000000e+00, 0.0000000e+00]
# sql_feature_encode_matrix2=  [0, 0 ,1, 0]


# operators_env_matrix3=  [1, 1, 1, 1, 1, 1, 1, 3, 4, 4, 4, 3, 4, 3, 4]
# indexes_env_matrix3 = [1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 12, 5, 6, 11, 6, 7, 10, 7, 8, 9, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0]
# query_enc_matrix3 = [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.7872569e-04, 
# 0.0000000e+00, 0.0000000e+00, 5.0000000e-01, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 2.2359693e-05, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 5.1583308e-01, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 1.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
# 0.0000000e+00, 0.0000000e+00, 9.0607846e-01, 0.0000000e+00, 0.0000000e+00]
# sql_feature_encode_matrix3= [0, 0, 1, 0]

# operators_env_matrix4 = [1, 1, 1, 0, 1, 1, 0, 3, 3, 4, 4, 3, 4, 4, 4]
# indexes_env_matrix4 = [1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 12, 5, 6, 11, 6, 7, 10, 7, 8, 9, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0]
# query_enc_matrix4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36103013, 0.0, 0.0, 0.25, 0.0, 0.0, 0.00884956, 0.00884956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.01711578, 0.03648241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17573306, 0.0, 0.0]
# sql_feature_encode_matrix4 = [0, 0, 1, 0]


# operators_env_matrix5= [1, 0, 1, 0, 1, 1, 0, 3, 3, 4, 4, 3, 4, 3, 4]
# indexes_env_matrix5= [1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 12, 5, 6, 11, 6, 7, 10, 7, 8, 9, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0]
# query_enc_matrix5= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36103013, 0.0, 0.0, 0.25, 0.0, 0.0, 0.00884956, 0.00884956, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.02043918, 0.13961385, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4102927, 0.0, 0.0]
# sql_feature_encode_matrix5= [0, 0, 1, 0]



# operators_env_matrix6= [0, 1, 1, 0, 1, 0, 1, 0, 3, 3, 4, 3, 4, 3, 4, 4, 3]
# indexes_env_matrix6= [1, 2, 17, 2, 3, 16, 3, 4, 15, 4, 5, 14, 5, 6, 13, 6, 7, 12, 7, 8, 11, 8, 9, 10, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0]
# query_enc_matrix6= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04233246, 0.0, 0.0, 0.25, 0.0, 0.00884956, 0.0, 0.00884956, 0.0, 0.0, 0.14285715, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
# sql_feature_encode_matrix6= [0, 0, 1, 0]


# operators_env_matrix7= [1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 4, 4, 4, 4, 4, 4, 4]
# indexes_env_matrix7= [1, 2, 17, 2, 3, 16, 3, 4, 15, 4, 5, 14, 5, 6, 13, 6, 7, 12, 7, 8, 11, 8, 9, 10, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0]
# query_enc_matrix7= [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.6103013e-01, 0.0000000e+00, 0.0000000e+00, 2.5000000e-01, 0.0000000e+00, 8.8495575e-03, 0.0000000e+00, 8.8495575e-03, 0.0000000e+00, 0.0000000e+00, 1.4285715e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.8589478e-04, 0.0000000e+00, 0.0000000e+00]
# sql_feature_encode_matrix7= [0, 0, 1, 0]

# it seems that iteration is faster than vectorization, the matrix is small
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
       
  
 
# 存储所有矩阵的列表
# operators_env_matrices = [operators_env_matrix1, operators_env_matrix2, operators_env_matrix3, operators_env_matrix4, operators_env_matrix5, operators_env_matrix6, operators_env_matrix7]
# indexes_env_matrices = [indexes_env_matrix1, indexes_env_matrix2, indexes_env_matrix3, indexes_env_matrix4, indexes_env_matrix5, indexes_env_matrix6, indexes_env_matrix7]
# query_enc_matrices = [query_enc_matrix1, query_enc_matrix2, query_enc_matrix3, query_enc_matrix4, query_enc_matrix5, query_enc_matrix6, query_enc_matrix7]
# sql_feature_encode_matrices = [sql_feature_encode_matrix1, sql_feature_encode_matrix2, sql_feature_encode_matrix3, sql_feature_encode_matrix4, sql_feature_encode_matrix5, sql_feature_encode_matrix6, sql_feature_encode_matrix7]

# # 循环比较矩阵
# for i in range(len(operators_env_matrices)):
#     for j in range(i + 1, len(operators_env_matrices)):
#         difference_operators = calculate_difference(operators_env_matrices[i], operators_env_matrices[j])
#         difference_indexes = calculate_difference(indexes_env_matrices[i], indexes_env_matrices[j])
#         difference_sql_feature = calculate_difference(sql_feature_encode_matrices[i], sql_feature_encode_matrices[j])
#         l2_distance = np.linalg.norm(np.array(query_enc_matrices[i]) - np.array(query_enc_matrices[j]))

#         # 输出结果
#         print(f"Comparison between matrix {i + 1} and matrix {j + 1}:")
#         print("Difference in operators_env_matrix:", difference_operators)
#         print("Difference in indexes_env_matrix:", difference_indexes)
#         print("Difference in sql_feature_encode_matrix:", difference_sql_feature)
#         print("L2 distance in query_enc_matrix:", l2_distance)
#         print("Final value:", get_final_value(difference_operators, difference_indexes, difference_sql_feature, l2_distance))
#         print()