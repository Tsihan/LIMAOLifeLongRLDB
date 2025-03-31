import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
USE_BAO = True
PG_CONNECTION_STR = "dbname=imdbload user=qihan host=localhost port=5432"
EPISODE_LEN = 10
PROGRESS_CFG = "/mydata/LIMAOLifeLongRLDB/bao_server/current_progress.cfg"

def update_progress(iteration, episode):
    """write the current progress to a file"""
    # Check if the directory exists, if not, create it
    with open(PROGRESS_CFG, "w") as f:
        f.write(f"iteration={iteration}\n")
        f.write(f"episode={episode}\n")
    
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

def run_query(sql, bao_select=False, bao_reward=False):
    start = time()
    conn = psycopg2.connect(PG_CONNECTION_STR)
    try:
        cur = conn.cursor()
        cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
        cur.execute(f"SET enable_bao_selection TO {bao_select}")
        cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
        cur.execute("SET bao_num_arms TO 49")
        cur.execute("SET statement_timeout TO 10000")
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
        return 30  # 返回超时时间（30秒）
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

# Assuming the directory containing SQL files is provided as the first argument
query_directory = "/mydata/LIMAOLifeLongRLDB/imdb_small"
queries = get_all_queries_from_directory(query_directory)

print("Read", len(queries), "queries.")
print("Using Bao:", USE_BAO)

print("Executing queries using PG optimizer")
update_progress(0, 0)

for fp, q in queries:
    pg_time = run_query(q, bao_reward=True)
    print("x", "x", time(), fp, pg_time, "PG", flush=True)


for i in range(1, 3):
    # 每个新的 iteration，更新 cfg 文件，episode 先置为 0
    update_progress(i, 0)
    print(f"===Executing queries using BAO optimizer, iteration {i}===")
    if USE_BAO:
        os.system(f"cd bao_server && python3 baoctl.py --retrain")
        os.system("sync")
        # 按 EPISODE_LEN 将 queries 分割成若干部分
        num_episodes = (len(queries) + EPISODE_LEN - 1) // EPISODE_LEN
        for j in range(0, len(queries), EPISODE_LEN):
            # 计算当前 episode 序号（从 1 开始）
            current_episode = j // EPISODE_LEN + 1
            # 更新 cfg 文件中的 episode 信息
            update_progress(i, current_episode)
            queries_part = queries[j:j + EPISODE_LEN]
            print("Executing", len(queries_part), "queries using BAO optimizer")
            print(f"Episode {current_episode}, iteration {i}")
            for fp, q in queries_part:
                q_time = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO)
                print("BAO", time(), fp, q_time, flush=True)
# 在程序结束时可选地发送邮件通知
# send_email("Bao Experiment", "The experiment of IMDB static finished!", "2453939195@qq.com")
