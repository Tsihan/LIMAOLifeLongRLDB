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
PG_CONNECTION_STR_1 = "dbname=imdbload user=qihanzha host=localhost port=5438"

PG_CONNECTION_STR_3 = "dbname=tpch10load user=qihanzha host=localhost port=5438"
PG_CONNECTION_STR_5 = "dbname=soload user=qihanzha host=localhost port=5438"
TIME_OUT_IMDB = 10000
TIME_OUT_TPCH = 30000
TIME_OUT_STACK = 30000
TOTOAL_ITER = 200
NUM_PHASE = 40
query_directory_imdb_list = ["/home/qihanzha/LIMAOLifeLongRLDB/imdb_assorted_3", "/home/qihanzha/LIMAOLifeLongRLDB/imdb_assorted_4"]
query_directory_stack_list = ["/home/qihanzha/LIMAOLifeLongRLDB/so_assorted", "/home/qihanzha/LIMAOLifeLongRLDB/so_assorted_2"]
query_directory_tpch_list = ["/home/qihanzha/LIMAOLifeLongRLDB/tpch_assorted", "/home/qihanzha/LIMAOLifeLongRLDB/tpch_assorted_2", "/home/qihanzha/LIMAOLifeLongRLDB/tpch_assorted_3"]
PG_CONNECTION_STR_LIST = [PG_CONNECTION_STR_1, PG_CONNECTION_STR_3, PG_CONNECTION_STR_5]
# init_query_directory = "/home/qihanzha/LIMAOLifeLongRLDB/imdb_assorted_3"
init_query_directory_list = query_directory_imdb_list + query_directory_stack_list + query_directory_tpch_list

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

for init_query_directory in init_query_directory_list:
    init_queries = get_all_queries_from_directory(init_query_directory)
    print("Init query directory:", init_query_directory)
    print("Read", len(init_queries), "queries.")
    print("Executing init queries using PG optimizer")
    if "imdb_" in init_query_directory:
        pg_connection_str = PG_CONNECTION_STR_1
        timeout = TIME_OUT_IMDB
    elif "so_" in init_query_directory:
        pg_connection_str = PG_CONNECTION_STR_5
        timeout = TIME_OUT_STACK
    else:
        pg_connection_str = PG_CONNECTION_STR_3
        timeout = TIME_OUT_TPCH
    for fp, q in init_queries:
        pg_time = run_query(q, pg_connection_str, timeout, bao_reward=True)
        print("x", "x", fp, pg_time, "PG", flush=True)

