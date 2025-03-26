import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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

query_directory_imdb_list = ["/mydata/LIMAOLifeLongRLDB/imdb_assorted_3","/mydata/LIMAOLifeLongRLDB/imdb_assorted_4", "/mydata/LIMAOLifeLongRLDB/imdb_assorted_5"]
queries = get_all_queries_from_directory(query_directory)

print("Read", len(queries), "queries.")
print("Using Bao:", USE_BAO)

print("Executing queries using PG optimizer")

for fp, q in queries:
    pg_time = run_query(q, bao_reward=True)
    print("x", "x", time(), fp, pg_time, "PG", flush=True)


for i in range(50):
    print(f"===Executing queries using BAO optimizer, iteration {i+1}===")
    if USE_BAO:
        
        os.system("cd bao_server && python3 baoctl.py --retrain")
        os.system("sync")
        for fp, q in queries:
            q_time = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO)
            print("BAO", time(), fp, q_time, flush=True)
# 在程序结束时调用
send_email("Bao Experiment", "The experiment of IMDB static finished!","2453939195@qq.com")
