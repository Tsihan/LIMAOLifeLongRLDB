def parse_matrix(file_path):
    matrices = []
    current_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = [int(x) for x in line.split()]
            if numbers[0] == 1 and current_matrix:
                matrices.append(current_matrix)
                current_matrix = [numbers]
            else:
                current_matrix.append(numbers)
        if current_matrix:
            matrices.append(current_matrix)
    return matrices


def pad_matrices(matrices):
    max_length = max(len(matrix) for matrix in matrices)
    max_width = max(max(len(row) for row in matrix) for matrix in matrices)
    for matrix in matrices:
        for row in matrix:
            row.extend([0] * (max_width - len(row)))
        matrix.extend([[0] * max_width] * (max_length - len(matrix)))
    return matrices


def average_matrices(matrices):
    num_matrices = len(matrices)
    if num_matrices == 0:
        return []
    avg_matrix = []
    for i in range(len(matrices[0])):
        avg_row = []
        for j in range(len(matrices[0][0])):
            avg_row.append(sum(matrix[i][j] for matrix in matrices) / num_matrices)
        avg_matrix.append(avg_row)
    return avg_matrix


# Parse the matrices from the file
matrices = parse_matrix('indexes_env_matrix.txt')

# Pad the matrices to the same size
padded_matrices = pad_matrices(matrices)

# Calculate the average matrix
average_matrix = average_matrices(padded_matrices)

# Print the average matrix
# for row in average_matrix:
#     print(' '.join(f'{val:.2f}' for val in row))

average_matrix_rounded_flat = [round(val) for row in average_matrix for val in row]

print(average_matrix_rounded_flat)

# [1, 2, 15, 2, 3, 14, 3, 4, 13, 4, 5, 11, 5, 4, 10, 6, 5, 9, 7, 4, 6, 8, 4, 6, 2, 0, 0, 7,
#  3, 4, 8, 2, 3, 9, 2, 3, 9, 2, 2, 10, 1, 2, 9, 1, 1, 10, 1, 1, 7, 0, 0, 0, 0, 0, 8, 0, 0,
#  5, 0, 0, 6, 0, 0, 4, 0, 0, 5, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
#  1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]




