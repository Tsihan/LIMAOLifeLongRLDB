import argparse
import psycopg2
import os
from time import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import math
from datetime import datetime

# Parse command-line arguments for random seed
parser = argparse.ArgumentParser(description='Run BAO dynamic experiment.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

# Set the seed for random using command-line argument
random.seed(args.seed)

USE_BAO = True
PG_CONNECTION_STR_1 = "dbname=imdbload user=qihan host=localhost"
PG_CONNECTION_STR_2 = "dbname=imdbload_after2000 user=qihan host=localhost"
PG_CONNECTION_STR_3 = "dbname=tpch1load user=qihan host=localhost"
PG_CONNECTION_STR_4 = "dbname=tpch10load user=qihan host=localhost"
PG_CONNECTION_STR_5 = "dbname=soload user=qihan host=localhost"
TIME_OUT_IMDB = 32000
TIME_OUT_TPCH = 60000
TIME_OUT_STACK = 60000
EPISODE_LEN = 10
PROGRESS_CFG = "/mydata/LIMAOLifeLongRLDB/bao_server/current_progress.cfg"

TOTOAL_ITER = 200
NUM_PHASE = 40
query_directory_imdb_list = ["/mydata/LIMAOLifeLongRLDB/imdb_assorted_3", "/mydata/LIMAOLifeLongRLDB/imdb_assorted_4"]
query_directory_stack_list = ["/mydata/LIMAOLifeLongRLDB/so_assorted", "/mydata/LIMAOLifeLongRLDB/so_assorted_2"]
query_directory_tpch_list = ["/mydata/LIMAOLifeLongRLDB/tpch_assorted", "/mydata/LIMAOLifeLongRLDB/tpch_assorted_2", "/mydata/LIMAOLifeLongRLDB/tpch_assorted_3"]
PG_CONNECTION_STR_LIST = [PG_CONNECTION_STR_1, PG_CONNECTION_STR_2, PG_CONNECTION_STR_3, PG_CONNECTION_STR_4, PG_CONNECTION_STR_5]
init_query_directory = "/mydata/LIMAOLifeLongRLDB/imdb_assorted_3"
def update_progress(iteration, episode):
    """write the current progress to a file"""
    # Check if the directory exists, if not, create it
    with open(PROGRESS_CFG, "w") as f:
        f.write(f"iteration={iteration}\n")
        f.write(f"episode={episode}\n")
    
def random_partition(total, parts):
    # Calculate the lower and upper bound for each part.
    lower_bound = math.ceil(total / parts / 2)  # half of average rounded up
    upper_bound = math.floor(2 * total / parts)   # double of average rounded down

    result = []
    remaining_total = total
    remaining_parts = parts

    for i in range(parts):
        # Ensure the total sum is correct by adjusting bounds for the current part.
        current_lower = max(lower_bound, remaining_total - (remaining_parts - 1) * upper_bound)
        current_upper = min(upper_bound, remaining_total - (remaining_parts - 1) * lower_bound)
        value = random.randint(current_lower, current_upper)
        result.append(value)
        remaining_total -= value
        remaining_parts -= 1
    
    random.shuffle(result)
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

# Start of the experiment.
time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
send_email("Bao Experiment", f"The experiment of dynamic started! {time_now}", "2453939195@qq.com")
partitions = random_partition(TOTOAL_ITER, NUM_PHASE)
print("Partition:", partitions)
init_queries = get_all_queries_from_directory(init_query_directory)
print("Read", len(init_queries), "queries.")
print("Executing init queries using PG optimizer")

update_progress(0, 0)
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
        print("Cannot parse database name")
        raise Exception("Cannot parse database name")

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
        print("Unknown database type")
        raise Exception("Unknown database type")
        
    chosen_directory = random.choice(query_dirs)
    queries = get_all_queries_from_directory(chosen_directory)

    for i in range(partition):
        global_iter += 1
        update_progress(global_iter,0)
        print(f"=== Executing queries using Bao optimizer, global iteration {global_iter}/{TOTOAL_ITER}, partition {partition}, phase iteration {i+1} for database {dbname}, query directory {chosen_directory} ===")
        if USE_BAO:
            if i == 0:
                # drift!
                os.system("cd /mydata/LIMAOLifeLongRLDB/bao_server && python3 baoctl.py --retrain")
                os.system("sync")
            else:
                # normal, use the last iteration data to retrain
                # os.system(f"cd /mydata/LIMAOLifeLongRLDB/bao_server && python3 baoctl.py --retrain --iteration {global_iter-1}")
                # FIXME or we still use all data to retrain
                os.system(f"cd /mydata/LIMAOLifeLongRLDB/bao_server && python3 baoctl.py --retrain")
                os.system("sync")

            num_episodes = (len(queries) + EPISODE_LEN - 1) // EPISODE_LEN
            for j in range(0, len(queries), EPISODE_LEN):
                # 计算当前 episode 序号（从 1 开始）
                current_episode = j // EPISODE_LEN + 1
                # 更新 cfg 文件中的 episode 信息
                update_progress(global_iter, current_episode)
                queries_part = queries[j:j + EPISODE_LEN]
                print("Executing", len(queries_part), "queries using BAO optimizer")
                print(f"Global Iteration {global_iter}, Episode {current_episode}")
                for fp, q in queries_part:
                    q_time = run_query(q, PG_CONNECTION_STR, timeout, bao_reward=USE_BAO, bao_select=USE_BAO)
                    print("BAO", fp, q_time, flush=True)
                # light train
                os.system(f"cd /mydata/LIMAOLifeLongRLDB/bao_server && python3 baoctl.py --retrain --iteration {global_iter} --episode {current_episode}")
                os.system("sync")
        if global_iter % 10 == 0:
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_email("Bao Experiment", f"The experiment of chaos is currently at {global_iter}/{TOTOAL_ITER} iters! {time_now}", "2453939195@qq.com")
        
time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
send_email("Bao Experiment", f"The experiment of LIMAO_Bao chaos finished! {time_now}", "2453939195@qq.com")
