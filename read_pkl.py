import pickle

# 假设你的pkl文件路径是'path/to/your_file.pkl'
pkl_path = '/home/qihan/balsa_project/balsaLifeLongRLDB/data/initial_policy_data.pkl'

# 使用'rb'模式打开文件（读取二进制文件）
with open(pkl_path, 'rb') as file:
    data = pickle.load(file)

# 打印文件内容
print(data)

# 获取对象的所有属性和方法
attributes = dir(data)

# 过滤掉内置属性和方法，只保留自定义的属性
custom_attributes = [attr for attr in attributes if not attr.startswith('__')]

# 打印自定义属性及其值
for attr in custom_attributes:
    print(f"{attr}: {getattr(data, attr)}")