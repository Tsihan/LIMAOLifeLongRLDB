import psycopg2
import os
import sys
import random
from time import time, sleep

USE_BAO = True
PG_CONNECTION_STR = "dbname=imdb user=imdb host=localhost"

# https://stackoverflow.com/questions/312443/
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_query(sql, bao_select=False, bao_reward=False):
    start = time()
    conn = psycopg2.connect(PG_CONNECTION_STR)
    try:
        cur = conn.cursor()
        cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
        cur.execute(f"SET enable_bao_selection TO {bao_select}")
        cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
        cur.execute("SET bao_num_arms TO 49")
        cur.execute("SET statement_timeout TO 30000")
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


        
query_paths = sys.argv[1:]
queries = []
for fp in query_paths:
    with open(fp) as f:
        query = f.read()
    queries.append((fp, query))
print("Read", len(queries), "queries.")
print("Using Bao:", USE_BAO)

random.seed(42)
query_sequence = random.choices(queries, k=500)
pg_chunks, *bao_chunks = list(chunks(query_sequence, 25))

print("Executing queries using PG optimizer for initial training")

for fp, q in pg_chunks:
    pg_time = run_query(q, bao_reward=True)
    print("x", "x", time(), fp, pg_time, "PG", flush=True)

for c_idx, chunk in enumerate(bao_chunks):
    if USE_BAO:
        os.system("cd bao_server && python3 baoctl.py --retrain")
        os.system("sync")
    for q_idx, (fp, q) in enumerate(chunk):
        q_time = run_query(q, bao_reward=USE_BAO, bao_select=USE_BAO)
        print(c_idx, q_idx, time(), fp, q_time, flush=True)
