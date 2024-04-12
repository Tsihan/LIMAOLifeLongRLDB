def read_matrices(file_path):
    with open(file_path, 'r') as file:
        return [[int(num) for num in line.split()] for line in file]

def calculate_average_matrix(matrices):
    num_matrices = len(matrices)
    matrix_size = len(matrices[0])
    average_matrix = [sum(values) / num_matrices for values in zip(*matrices)]
    return [round(val) for val in average_matrix]

# 读取文件中的矩阵
matrices = read_matrices('sql_feature_encode_matrix.txt')

# 计算平均矩阵
average_matrix = calculate_average_matrix(matrices)

# 输出平均矩阵
print(average_matrix)
# [1, 1, 1, 0]