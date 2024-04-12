import numpy as np
from kmodes.kprototypes import KPrototypes
from itertools import chain

test_data_1_f1 = [1, 2, 21, 2, 3, 20, 3, 4, 19, 4, 5, 18, 5, 6, 17, 6, 7, 16, 7, 8, 15, 8, 9, 14, 9, 10, 13, 10, 11,
                  12, 11, 0, 0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18, 0, 0, 19, 0, 0, 20, 0, 0, 21, 0, 0]

test_data_1_f2 = [1, 1, 1, 1, 1, 1, 1, 0,
                  1, 1, 4, 4, 4, 3, 4, 4, 4, 3, 3, 4, 4]

test_data_1_f3 = [0, 0, 1, 0]

test_data_1_f4 = [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.84955749e-03, 8.84955749e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.04345236e-04, 4.28571433e-01,
                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.93923570e-04, 7.48666062e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 3.48630399e-01, 0.00000000e+00, 1.66666672e-01, 1.81086034e-01, 0.00000000e+00, 0.00000000e+00]

test_data_2_f1 = [1, 2, 13, 2, 3, 12, 3, 4, 11, 4, 5, 10, 5, 6, 9, 6, 7,
                  8, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0]

test_data_2_f2 = [1, 1, 1, 1, 1, 1, 3, 4, 5, 4, 4, 4, 4]

test_data_2_f3 = [0, 0, 1, 0]

test_data_2_f4 = [0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.6416635e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.4532309e-06, 0.0000000e+00,
                  0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00]

test_data_3_f1 = [1, 2, 13, 2, 3, 12, 3, 4, 11, 4, 5, 10, 5, 6, 9, 6, 7,
                  8, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0, 0, 12, 0, 0, 13, 0, 0]

test_data_3_f2 = [1, 1, 1, 1, 1, 1, 3, 4, 5, 4, 4, 4, 4]

test_data_3_f3 = [0, 0, 1, 0]

test_data_3_f4 = [0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.6416635e-01, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 7.4532309e-06, 0.0000000e+00,
                  0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00]




class Kproto_DataProcessor:
    def __init__(self, index_path, operator_path, sql_path, query_path, matrix_size):
        self.index_path = index_path
        self.operator_path = operator_path
        self.sql_path = sql_path
        self.query_path = query_path
        self.matrix_size = matrix_size
        self.kproto_hashjoin = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)
        self.kproto_nestedloop = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)
        self.kproto_other = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)
        self.max_lengths = [0, 0, 0, 0]
        self.categorical_indices = []
        self.load_data()

    def load_data(self):
        self.feature1_matrices = self.read_indexes_env_matrices(
            self.index_path)
        self.feature2_matrices = self.read_operators_env_matrices(
            self.operator_path)
        self.feature3_matrices = self.read_sql_feature_matrices(self.sql_path)
        self.feature4_matrices = self.read_query_enc_matrices(
            self.query_path, self.matrix_size)
        assert len(self.feature1_matrices) == len(self.feature2_matrices) == len(
            self.feature3_matrices) == len(self.feature4_matrices)

        self.split_transform_features()
        self.create_data_points()

    def read_indexes_env_matrices(self, file_path):
        # Implement this method based on your previous script
        matrices = []
        current_matrix = []
        with open(file_path, 'r') as file:
            for line in file:
                numbers = [int(x) for x in line.split()]
                if numbers[0] == 1 and current_matrix:
                    # 展平 current_matrix 并添加到 matrices
                    flat_matrix = [
                        item for sublist in current_matrix for item in sublist]
                    matrices.append(flat_matrix)
                    current_matrix = [numbers]
                else:
                    current_matrix.append(numbers)
            if current_matrix:
                # 展平最后一个 current_matrix 并添加到 matrices
                flat_matrix = [
                    item for sublist in current_matrix for item in sublist]
                matrices.append(flat_matrix)
        return matrices

    def read_operators_env_matrices(self, file_path):
        # Implement this method based on your previous script
        with open(file_path, 'r') as file:
            return [[int(num) for num in line.split()] for line in file]

    def read_sql_feature_matrices(self, file_path):
        # Implement this method based on your previous script
        with open(file_path, 'r') as file:
            return [[int(num) for num in line.split()] for line in file]

    def read_query_enc_matrices(self, file_path, matrix_size):
        # Implement this method based on your previous script
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

    def split_transform_features(self):
        # Implement this method based on your previous script
        # 拆分并转换 feature2_matrices
        self.feature2_matrix_hashjoin = [self.transform_matrix(
            matrix, {1, 2, 3, 4, 5}) for matrix in self.feature2_matrices]
        self.feature2_matrix_nestedloop = [self.transform_matrix(
            matrix, {0, 2, 3, 4, 5}) for matrix in self.feature2_matrices]
        self.feature2_matrix_other = [self.transform_matrix(
            matrix, {0, 1}) for matrix in self.feature2_matrices]

    def create_data_points(self):
        # Implement this method based on your previous script
        # 构建数据点
        data_points_hashjoin = []
        data_points_nestedloop = []
        data_points_other = []

        
        for i in range(len(self.feature1_matrices)):
            data_point_hashjoin = [
                self.feature1_matrices[i],
                self.feature2_matrix_hashjoin[i],
                self.feature3_matrices[i],
                self.feature4_matrices[i]
            ]
            data_point_nestedloop = [
                self.feature1_matrices[i],
                self.feature2_matrix_nestedloop[i],
                self.feature3_matrices[i],
                self.feature4_matrices[i]
            ]

            data_point_other = [
                self.feature1_matrices[i],
                self.feature2_matrix_other[i],
                self.feature3_matrices[i],
                self.feature4_matrices[i]
            ]

            self.max_lengths[0] = max(self.max_lengths[0], len(self.feature1_matrices[i]))
            self.max_lengths[1] = max(self.max_lengths[1], len(self.feature2_matrices[i]))
            self.max_lengths[2] = max(self.max_lengths[2], len(self.feature3_matrices[i]))
            self.max_lengths[3] = max(self.max_lengths[3], len(self.feature4_matrices[i]))
            data_points_hashjoin.append(data_point_hashjoin)
            data_points_nestedloop.append(data_point_nestedloop)
            data_points_other.append(data_point_other)

        # 为每个数据点的每个特征补齐缺失值

        for data_point in data_points_hashjoin:
            for i in range(4):
                if i < 3:  # 对于分类特征，用-1填充
                    data_point[i].extend(
                        [-1] * (self.max_lengths[i] - len(data_point[i])))
                else:  # 对于数值特征，用0填充
                    data_point[i].extend(
                        [0] * (self.max_lengths[i] - len(data_point[i])))

        for data_point in data_points_nestedloop:
            for i in range(4):
                if i < 3:  # 对于分类特征，用-1填充
                    data_point[i].extend(
                        [-1] * (self.max_lengths[i] - len(data_point[i])))
                else:  # 对于数值特征，用0填充
                    data_point[i].extend(
                        [0] * (self.max_lengths[i] - len(data_point[i])))

        for data_point in data_points_other:
            for i in range(4):
                if i < 3:  # 对于分类特征，用-1填充
                    data_point[i].extend(
                        [-1] * (self.max_lengths[i] - len(data_point[i])))
                else:  # 对于数值特征，用0填充
                    data_point[i].extend(
                        [0] * (self.max_lengths[i] - len(data_point[i])))

        # 将数据点合并到一个数据集中并转换为NumPy数组
        data_hashjoin = np.array([np.array(list(chain.from_iterable(
            data_point))) for data_point in data_points_hashjoin])
        data_nestedloop = np.array([np.array(list(chain.from_iterable(
            data_point))) for data_point in data_points_nestedloop])
        data_other = np.array([np.array(list(chain.from_iterable(
            data_point))) for data_point in data_points_other])
        # 创建 KPrototypes 实例，设置聚类数为4，使用'Huang'初始化方法
        self.kproto_hashjoin = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)
        self.kproto_nestedloop = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)
        self.kproto_other = KPrototypes(
            n_clusters=4, init='Huang', random_state=42)

        # 对数据进行聚类，分类特征的索引为前三个特征的索引
        self.categorical_indices = list(range(sum(self.max_lengths[:3])))
        self.kproto_hashjoin.fit_predict(
            data_hashjoin, categorical=self.categorical_indices)
        self.kproto_nestedloop.fit_predict(
            data_nestedloop, categorical=self.categorical_indices)
        self.kproto_other.fit_predict(
            data_other, categorical=self.categorical_indices)

    def transform_matrix(self, matrix, replace_values):
        matrix = np.array(matrix)
        mask = np.isin(matrix, list(replace_values))
        matrix[mask] = 6
        return matrix.tolist()


    def prepare_data_point(self, feature_list):
        prepared_data_point = []
        for i, feature in enumerate(feature_list):
            if len(feature) > self.max_lengths[i]:
                feature = feature[:self.max_lengths[i]]
            if i < 3:
                prepared_data_point.extend(
                    feature + [-1] * (self.max_lengths[i] - len(feature)))
            else:
                prepared_data_point.extend(
                    feature + [0] * (self.max_lengths[i] - len(feature)))
        return np.array(prepared_data_point)

    def predict_datapoint(self, data_point):
        # Implement prediction based on your previous script

        # 对新数据点的feature2应用转换规则
        test_data_1_f2_hashjoin = self.transform_matrix(
            data_point[1], {1, 2, 3, 4, 5})
        test_data_1_f2_nestedloop = self.transform_matrix(
            data_point[1], {0, 2, 3, 4, 5})
        test_data_1_f2_other = self.transform_matrix(data_point[1], {0, 1})


# 准备新的数据点
        test_data_point_1_hashjoin = self.prepare_data_point(
            [data_point[0], test_data_1_f2_hashjoin,
                data_point[2], data_point[3]]
        )
        test_data_point_1_nestedloop = self.prepare_data_point(
            [data_point[0], test_data_1_f2_nestedloop,
                data_point[2], data_point[3]]
        )
        test_data_point_1_other = self.prepare_data_point(
            [data_point[0], test_data_1_f2_other, data_point[2], data_point[3]]
        )


# 使用transformed数据点进行预测
        predicted_label_1_hashjoin = self.kproto_hashjoin.predict(
            np.array([test_data_point_1_hashjoin]), categorical=self.categorical_indices)
        predicted_label_1_nestedloop = self.kproto_nestedloop.predict(
            np.array([test_data_point_1_nestedloop]), categorical=self.categorical_indices)
        predicted_label_1_other = self.kproto_other.predict(
            np.array([test_data_point_1_other]), categorical=self.categorical_indices)

        # 打印预测结果
        print('Test Data Point 1 - Hash Join Cluster:',
              predicted_label_1_hashjoin)
        print('Test Data Point 1 - Nested Loop Cluster:',
              predicted_label_1_nestedloop)
        print('Test Data Point 1 - other Cluster:', predicted_label_1_other)




# Example usage:
processor = Kproto_DataProcessor(
    index_path='/home/qihan/balsaLifeLongRLDB/balsa/deal_assorted_text/indexes_env_matrix.txt',
    operator_path='/home/qihan/balsaLifeLongRLDB/balsa/deal_assorted_text/operators_env_matrix.txt',
    sql_path='/home/qihan/balsaLifeLongRLDB/balsa/deal_assorted_text/sql_feature_encode_matrix.txt',
    query_path='/home/qihan/balsaLifeLongRLDB/balsa/deal_assorted_text/query_enc_matrix.txt',
    matrix_size=46
)
# 获取当前执行预测所用时间
import time
start = time.time()
processor.predict_datapoint([test_data_1_f1, test_data_1_f2, test_data_1_f3, test_data_1_f4])
end = time.time()
print("Time taken for prediction:", end - start)

start = time.time()
processor.predict_datapoint([test_data_2_f1, test_data_2_f2, test_data_2_f3, test_data_2_f4])
end = time.time()
print("Time taken for prediction:", end - start)

start = time.time()
processor.predict_datapoint([test_data_3_f1, test_data_3_f2, test_data_3_f3, test_data_3_f4])
end = time.time()
print("Time taken for prediction:", end - start)
