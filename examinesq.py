import os
import psycopg2
import time

# 数据库连接参数
db_params = {
    'database': 'imdbload',
    'user': 'qihan',
    'host': 'localhost',
    'port': '5432'
}

def execute_sql_file(file_path, conn):
    with open(file_path, 'r') as file:
        sql = file.read()

    cursor = conn.cursor()
    try:
        cursor.execute("SET statement_timeout TO 30000")  # 设置超时时间为 30 秒
        start_time = time.time()
        cursor.execute(sql)
        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            print(f"Execution of {file_path} was aborted after 30 seconds.")
        else:
            print(f"Execution of {file_path} completed in {elapsed_time} seconds.")
    except psycopg2.OperationalError as e:
        if "timeout expired" in str(e):
            print(f"Execution of {file_path} was aborted due to timeout.")
        else:
            print(f"An error occurred while executing {file_path}: {e}")
        conn.rollback()  # 回滚当前事务
    except Exception as e:
        print(f"An error occurred while executing {file_path}: {e}")
        conn.rollback()  # 回滚当前事务
    finally:
        cursor.close()

def main(sql_directory):
    conn = psycopg2.connect(**db_params)
    for file_name in os.listdir(sql_directory):
        if file_name.endswith('.sql'):
            file_path = os.path.join(sql_directory, file_name)
            execute_sql_file(file_path, conn)
    conn.close()

if __name__ == '__main__':
    sql_directory = '/home/qihan/balsaLifeLongRLDB/queries/job_changed'
    main(sql_directory)
