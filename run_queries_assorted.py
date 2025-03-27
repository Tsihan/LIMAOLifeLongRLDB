import psycopg2
import os
import sys
from time import time, sleep
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import math
from datetime import datetime



# set the seed for random
random.seed(42)
USE_BAO = True
PG_CONNECTION_STR_1 = "dbname=imdbload user=qihan host=localhost"
PG_CONNECTION_STR_2 = "dbname=imdbload_after2000 user=qihan host=localhost"
PG_CONNECTION_STR_3 = "dbname=tpch1load user=qihan host=localhost"
PG_CONNECTION_STR_4 = "dbname=tpch10load user=qihan host=localhost"
PG_CONNECTION_STR_5 = "dbname=soload user=qihan host=localhost"
# PG_CONNECTION_STR_6 = "dbname=soloaddownsize user=qihan host=localhost"
TIME_OUT_IMDB = 30000
TIME_OUT_TPCH = 30000
TIME_OUT_STACK = 30000
TOTOAL_ITER = 100
NUM_PHASE = 10
query_directory_imdb_list = ["/mydata/LIMAOLifeLongRLDB/imdb_assorted_3","/mydata/LIMAOLifeLongRLDB/imdb_assorted_4"]
query_directory_stack_list = ["/mydata/LIMAOLifeLongRLDB/so_assorted","/mydata/LIMAOLifeLongRLDB/so_assorted_2"]
query_directory_tpch_list = ["/mydata/LIMAOLifeLongRLDB/tpch_assorted","/mydata/LIMAOLifeLongRLDB/tpch_assorted_2"]
PG_CONNECTION_STR_LIST = [PG_CONNECTION_STR_1, PG_CONNECTION_STR_2, PG_CONNECTION_STR_3, PG_CONNECTION_STR_4, PG_CONNECTION_STR_5]
init_query_directory = "/mydata/LIMAOLifeLongRLDB/imdb_assorted_3"



def random_partition(total, parts):
    # calculate the lower and upper bound for each part
    lower_bound = math.ceil(total / parts / 2)  # total/parts's half rounded up
    upper_bound = math.floor(2 * total / parts)  # total/parts's double rounded down

    result = []
    remaining_total = total
    remaining_parts = parts

    for i in range(parts):
        # to ensure the total sum is correct, we need to calculate the current part's
        # lower and upper bound
        # current part's lower bound cannot be lower than: total - (remaining_parts-1)*upper_bound
        # current part's lower bound cannot be lower than: lower_bound
        current_lower = max(lower_bound, remaining_total - (remaining_parts - 1) * upper_bound)
        # in the same way, current part's upper bound cannot be higher than: total - (remaining_parts-1)*lower_bound
        # current part's upper bound cannot be higher than: upper_bound
        current_upper = min(upper_bound, remaining_total - (remaining_parts - 1) * lower_bound)

        # choose a random value between current_lower and current_upper
        value = random.randint(current_lower, current_upper)
        result.append(value)

        remaining_total -= value
        remaining_parts -= 1

    return result

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
        conn.commit()
        conn.autocommit = True
        cur.execute('DISCARD ALL;')
    except psycopg2.extensions.QueryCanceledError:

        try:
            conn.rollback()
            conn.autocommit = True
            cur.execute('DISCARD ALL;')
        except Exception:
            pass
        conn.close()
        return timeout / 1000  
    except Exception as e:
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
                queries.append((file, query))
    return queries


time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email("Bao Experiment", f"The experiment of dynamic started! {time_now}","2453939195@qq.com")
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
    PG_CONNECTION_STR = random.choice(PG_CONNECTION_STR_LIST)
    
    dbname = None
    for part in PG_CONNECTION_STR.split():
        if part.startswith("dbname="):
            dbname = part.split("=")[1]
            break
    if not dbname:
        print("cannot parse database name")
        raise Exception("cannot parse database name")

    if "imdb" in dbname:
        query_dirs = query_directory_imdb_list
        timeout = TIME_OUT_IMDB
    elif "tpch" in dbname:
        query_dirs = query_directory_tpch_list
        timeout = TIME_OUT_TPCH
    elif "soload" in dbname:
        query_dirs = query_directory_stack_list
        timeout = TIME_OUT_STACK
    else:
        print("unknown database type")
        raise Exception("unknown database type")
    chosen_directory = random.choice(query_dirs)
    queries = get_all_queries_from_directory(chosen_directory)

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
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_email("Bao Experiment", f"The experiment of dynamic is current in {global_iter} iters! {time_now}","2453939195@qq.com")
        
time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
send_email("Bao Experiment", f"The experiment of dynamic finished! {time_now}","2453939195@qq.com")
