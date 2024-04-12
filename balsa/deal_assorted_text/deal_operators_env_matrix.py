def read_matrices(file_path):
    with open(file_path, 'r') as file:
        return [[int(num) for num in line.split()] for line in file]


def extend_matrices(matrices, max_length):
    for matrix in matrices:
        matrix.extend([0] * (max_length - len(matrix)))


def calculate_average_matrix(matrices):
    max_length = max(len(matrix) for matrix in matrices)
    extend_matrices(matrices, max_length)
    average_matrix = [round(sum(values) / len(values)) for values in zip(*matrices)]
    return average_matrix


# 读取文件中的矩阵
matrices = read_matrices('operators_env_matrix.txt')

# 计算平均矩阵
average_matrix = calculate_average_matrix(matrices)
#[1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 输出平均矩阵
print(average_matrix)
