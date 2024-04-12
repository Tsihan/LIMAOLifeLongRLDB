import re

# 定义输入文件和输出文件的路径
input_file_path = 'imdb_aasorted.txt'
output_files = {
    'operators_env_matrix': 'operators_env_matrix.txt',
    'indexes_env_matrix': 'indexes_env_matrix.txt',
    'query_enc_matrix': 'query_enc_matrix.txt',
    'sql_feature_encode_matrix': 'sql_feature_encode_matrix.txt'
}

# 初始化字典来存储每个矩阵的内容
matrices = {key: [] for key in output_files.keys()}

# 读取输入文件并提取矩阵
current_matrix = None
with open(input_file_path, 'r') as file:
    for line in file:
        # 检查当前行是否包含矩阵的开始
        for key in matrices.keys():
            if key in line and 'for this sql' in line:
                current_matrix = key
                break

        # 如果当前处于矩阵的读取状态
        if current_matrix:
            # 提取矩阵内容，包括科学计数法
            content = re.findall(r'[\d.-]+(?:e[+-]?\d+)?', line)
            if content:
                matrices[current_matrix].append(' '.join(content) + '\n')

            # 检查当前行是否包含矩阵的结束
            if ']' in line:
                current_matrix = None

# 将提取的矩阵写入对应的输出文件
for key, file_path in output_files.items():
    with open(file_path, 'w') as file:
        file.writelines(matrices[key])
