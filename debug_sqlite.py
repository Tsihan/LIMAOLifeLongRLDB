import sqlite3

def print_all_tables_and_records(db_path):
    try:
        # 连接到 SQLite 数据库文件
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取数据库中所有表的名称
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("数据库中没有表。")
            return

        # 对每个表打印表名以及前10条记录
        for table in tables:
            table_name = table[0]
            print(f"表名: {table_name}")
            # 查询表中的前10条记录
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
            print("-" * 40)

    except Exception as e:
        print("发生错误：", e)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # 修改这里的路径为你的 SQLite 数据库文件路径
    db_path = "/mydata/LIMAOLifeLongRLDB/bao_server/bao.db"
    print_all_tables_and_records(db_path)