import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
USE_BAO = True
PG_CONNECTION_STR = "dbname=imdbload user=qihan host=localhost port=5438"
def send_email(subject, body, to_email):
    from_email = "xxx"
    password = "xxx"

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
    while True:
        try:
            conn = psycopg2.connect(PG_CONNECTION_STR)
            cur = conn.cursor()
            cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
            cur.execute(f"SET enable_bao_selection TO {bao_select}")
            cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
            # cur.execute("SET bao_num_arms TO 5")
            cur.execute("SET bao_num_arms TO 49")
            cur.execute("SET statement_timeout TO 10000")
            cur.execute(sql)
            cur.fetchall()
            cur.execute('DISCARD ALL;')
            conn.close()
            break
        except:
            sleep(1)
            continue
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
query_directory = "/mydata/BaoForPostgreSQL/imdb_assorted_3"
queries = get_all_queries_from_directory(query_directory)

print("Read", len(queries), "queries.")
print("Using Bao:", USE_BAO)

print("Executing queries using PG optimizer")

for fp, q in queries:
    pg_time = run_query(q, bao_reward=True)
    print("x", "x", time(), fp, pg_time, "PG", flush=True)


for i in range(50):
    print(f"Executing queries using BAO optimizer, iteration {i+1}")
    if USE_BAO:
        
        os.system("cd bao_server && python3 baoctl.py --retrain")
        os.system("sync")
        for fp, q in queries:
            q_time = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO)
            print("BAO", time(), fp, q_time, flush=True)
# 在程序结束时调用
send_email("Bao Experiment", "The experiment of IMDB XXX finished!", "EMAIL")
