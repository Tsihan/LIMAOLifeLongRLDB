import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random

# set the seed for random
random.seed(42)
USE_BAO = True
PG_CONNECTION_STR_1 = "dbname=imdbload user=qihan host=localhost"
PG_CONNECTION_STR_2 = "dbname=imdbload_after2000 user=qihan host=localhost"
PG_CONNECTION_STR_3 = "dbname=tpch1load user=qihan host=localhost"
PG_CONNECTION_STR_4 = "dbname=tpch10load user=qihan host=localhost"
PG_CONNECTION_STR_5 = "dbname=soload user=qihan host=localhost"
PG_CONNECTION_STR_6 = "dbname=soloaddownsize user=qihan host=localhost"
TIME_OUT_IMDB = 30000
TIME_OUT_TPCH = 30000
TIME_OUT_STACK = 30000
TOTOAL_ITER = 100
NUM_PHASE = 10
query_directory_imdb_list = ["/mydata/LIMAOLifeLongRLDB/imdb_assorted_3","/mydata/LIMAOLifeLongRLDB/imdb_assorted_4"]
query_directory_stack_list = ["/mydata/LIMAOLifeLongRLDB/so_assorted","/mydata/LIMAOLifeLongRLDB/so_assorted_2"]
query_directory_tpch_list = ["/mydata/LIMAOLifeLongRLDB/tpch_assorted","/mydata/LIMAOLifeLongRLDB/tpch_assorted_2"]
PG_CONNECTION_STR_LIST = [PG_CONNECTION_STR_1, PG_CONNECTION_STR_2, PG_CONNECTION_STR_3, PG_CONNECTION_STR_4, PG_CONNECTION_STR_5, PG_CONNECTION_STR_6]
init_query_directory = "/mydata/LIMAOLifeLongRLDB/imdb_assorted_3"



def random_partition(total, parts):
    # 生成 parts-1 个切割点，将 [1, total-1] 划成 parts 段
    cuts = sorted(random.sample(range(1, total), parts - 1))
    cuts = [0] + cuts + [total]
    return [cuts[i+1] - cuts[i] for i in range(parts)]

def send_email(subject, body, to_email):
    from_email = "2453939195@qq.com"
    password = "bajbveysllkjdjbd"
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.qq.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_query(sql, connection_str, timeout, bao_select=False, bao_reward=False):
    start = time()
    conn = psycopg2.connect(connection_str)
    try:
        cur = conn.cursor()
        cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
        cur.execute(f"SET enable_bao_selection TO {bao_select}")
        cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
        cur.execute("SET bao_num_arms TO 49")
        cur.execute(f"SET statement_timeout TO {timeout}")
        cur.execute(sql)
        cur.fetchall()
        # 提交事务，结束事务块
        conn.commit()
        # 切换为自动提交模式，以便执行 DISCARD ALL 不在事务块中
        conn.autocommit = True
        cur.execute('DISCARD ALL;')
    except psycopg2.extensions.QueryCanceledError:
        # 对于超时异常，先回滚，结束事务块
        try:
            conn.rollback()
            conn.autocommit = True
            cur.execute('DISCARD ALL;')
        except Exception:
            pass
        conn.close()
        return timeout / 1000  # 返回超时时间（30秒）
    except Exception as e:
        # 其他异常先回滚，再抛出错误
        conn.rollback()
        conn.close()
        raise e
    conn.close()
    stop = time()
    return stop - start

def get_all_queries_from_directory(directory):
    queries = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sql"):
                fp = os.path.join(root, file)
                with open(fp) as f:
                    query = f.read()
                # 只保留文件名
                queries.append((file, query))
    return queries


send_email("Bao Experiment", "The experiment of dynamic started!","2453939195@qq.com")
partitions = random_partition(TOTOAL_ITER, NUM_PHASE)
print("Partition:", partitions)
init_queries = get_all_queries_from_directory(init_query_directory)
print("Read", len(init_queries), "queries.")
print("Executing init queries using PG optimizer")
for fp, q in init_queries:
    pg_time = run_query(q, PG_CONNECTION_STR_1, TIME_OUT_IMDB, bao_reward=True)
    print("x", "x", fp, pg_time, "PG", flush=True)

global_iter = 0            
for partition in partitions:
    # 随机选择一个 PG 连接字符串
    PG_CONNECTION_STR = random.choice(PG_CONNECTION_STR_LIST)
    
    # 从连接字符串中解析数据库名称
    dbname = None
    for part in PG_CONNECTION_STR.split():
        if part.startswith("dbname="):
            dbname = part.split("=")[1]
            break
    if not dbname:
        print("无法解析数据库名称，跳过当前 phase")
        raise Exception("无法解析数据库名称")

    # 根据数据库名称选择对应的 workload 目录列表和超时设置
    if "imdb" in dbname:
        query_dirs = query_directory_imdb_list
        timeout = TIME_OUT_IMDB
    elif "tpch" in dbname:
        query_dirs = query_directory_tpch_list
        timeout = TIME_OUT_TPCH
    elif "solo" in dbname:
        query_dirs = query_directory_stack_list
        timeout = TIME_OUT_STACK
    else:
        print("未知数据库类型:", dbname)
        raise Exception("未知数据库类型")
    
    chosen_directory = random.choice(query_dirs)
    queries = get_all_queries_from_directory(chosen_directory)

    # 针对当前 phase 的每一次迭代
    for i in range(partition):
        global_iter += 1
        print(f"===Executing queries using Bao optimizer, global iteration {global_iter}/100, \
              partition {partition}, phase iteration {i+1} for database {dbname}, query directory {chosen_directory}===")
        if USE_BAO:
            os.system("cd bao_server && python3 baoctl.py --retrain")
            os.system("sync")
            for fp, q in queries:
                q_time = run_query(q, PG_CONNECTION_STR, timeout, bao_reward=USE_BAO, bao_select=USE_BAO)
                print("BAO", fp, q_time, flush=True)
        if global_iter % 10 == 0:
            send_email("Bao Experiment", f"The experiment of dynamic is current in {global_iter} iters!","2453939195@qq.com")
        
        
send_email("Bao Experiment", "The experiment of dynamic finished!","2453939195@qq.com")
