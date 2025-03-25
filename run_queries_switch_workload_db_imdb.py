import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
USE_BAO = True

PG_CONNECTION_STR_origin = "dbname=imdbload user=qihan host=localhost"
PG_CONNECTION_STR_second = "dbname=imdbload_after2000 user=qihan host=localhost"
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

def run_query(sql, connect_string, bao_select=False, bao_reward=False):
    start = time()
    while True:
        try:
            conn = psycopg2.connect(connect_string)
            cur = conn.cursor()
            cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
            cur.execute(f"SET enable_bao_selection TO {bao_select}")
            cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
            cur.execute("SET bao_num_arms TO 5")
            cur.execute("SET statement_timeout TO 512000")
            cur.execute(sql)
            cur.fetchall()
            conn.close()
            break
        except Exception as e:
            return 512  # Return 512 seconds as the time recorded
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
                queries.append((fp, query))
    return queries



query_directory_1, query_directory_2 = "/mydata/BaoForPostgreSQL/imdb_assorted_3","/mydata/BaoForPostgreSQL/imdb_assorted_4"

queries_assorted_3 = get_all_queries_from_directory(query_directory_1)
queries_assorted_4 = get_all_queries_from_directory(query_directory_2)

print("Read", len(queries_assorted_3), "queries from", query_directory_1)
print("Read", len(queries_assorted_4), "queries from", query_directory_2)
print("Using Bao:", USE_BAO)

print("Executing queries using PG optimizer")

for fp, q in queries_assorted_3:
    pg_time = run_query(q, PG_CONNECTION_STR_origin,bao_reward=True)
    print("x", "x", time(), fp, pg_time, "PG", flush=True)

use_assorted_3 = True
use_origin_db = True

for i in range(100):
    print(f"Executing queries using BAO optimizer, iteration {i+1}")
    if i % 5 == 0 and i != 0:
        use_assorted_3 = not use_assorted_3
        use_origin_db = not use_origin_db

    chosen_queries = queries_assorted_3 if use_assorted_3 else queries_assorted_4
    chosen_connect_string = PG_CONNECTION_STR_origin if use_origin_db else PG_CONNECTION_STR_second
    
    print("Using assorted queries from:", "query_directory_1" if use_assorted_3 else "query_directory_2")
    if USE_BAO:
        
        os.system("cd bao_server && python3 baoctl.py --retrain")
        os.system("sync")
        for fp, q in chosen_queries:
            q_time = run_query(q, chosen_connect_string,bao_reward=USE_BAO, bao_select=USE_BAO)
            print("BAO", time(), fp, q_time, flush=True)
# 在程序结束时调用
send_email("Bao Experiment", "The experiment of IMDB both switching workload and db finished!", "EMAIL")
