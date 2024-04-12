def read_matrices(file_path, matrix_size):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrices = []
        current_matrix = []
        for line in lines:
            numbers = [float(num) for num in line.split()]
            current_matrix.extend(numbers)
            if len(current_matrix) == matrix_size:
                matrices.append(current_matrix)
                current_matrix = []
        if current_matrix:
            matrices.append(current_matrix)
    return matrices


def calculate_average_matrix(matrices, matrix_size):
    sum_matrix = [0] * matrix_size
    for matrix in matrices:
        for i in range(len(matrix)):
            sum_matrix[i] += matrix[i]
    average_matrix = [sum_value / len(matrices) for sum_value in sum_matrix]
    return average_matrix


matrix_size = 46
matrices = read_matrices('query_enc_matrix.txt', matrix_size)
average_matrix = calculate_average_matrix(matrices, matrix_size)
print(average_matrix)
# [0.010526315789473684, 0.17166938674736842, 0.021052631578947368, 0.010526315789473684,
#  0.031578947368421054, 0.2388503383557894, 0.12800837582379948, 0.05, 0.039473684210526314,
#  0.19318683248483579, 0.003948404210526316, 0.021052631578947368, 0.2, 0.16842105263157894,
#  0.03344201234189474, 0.0027014440789473676, 0.002701444105263157, 0.00018630647368421055,
#  0.0631753902157956, 0.039097746305263165, 0.003007518947368421, 0.003007518947368421,
#  0.05204678372631577, 0.4522579790700315, 0.021052631578947368, 0.021052631578947368,
#  0.08108109977885265, 0.22141186853052625, 0.021052631578947368, 0.010278366315789473,
#  0.042105263157894736, 0.6210526315789474, 0.16842105263157894, 0.1525861300823684,
#  0.021265262315789474, 0.021349497372302843, 0.01491228151157895, 0.5011503272083473,
#  0.042105263157894736, 0.02336705694736842]


