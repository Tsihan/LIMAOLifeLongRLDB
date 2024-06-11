import os
import random

def select_files(directory, seed=52, fraction=1/6):
    files = [f for f in os.listdir(directory) if f.endswith('.sql')]
    random.seed(seed)
    selected_files = random.sample(files, int(len(files) * fraction))
    return selected_files

if __name__ == '__main__':
    directory = '/home/qihan/balsaLifeLongRLDB/queries/tpch_assorted_2'  # 当前目录
    selected_files = select_files(directory)
    print("Selected files:", selected_files)