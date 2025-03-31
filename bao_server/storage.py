import sqlite3
import json
import itertools

from common import BaoException
ROW_LIMIT = 500
CFG_FILE_PATH = "/mydata/LIMAOLifeLongRLDB/bao_server/current_progress.cfg"

def read_progress(cfg_file=CFG_FILE_PATH):
    """
    读取当前进度配置文件，返回 iteration 和 episode（如果读取失败，则默认返回0, 0）
    文件格式假定为：
        iteration=数字
        episode=数字
    """
    iteration = 0
    episode = 0
    try:
        with open(cfg_file, "r") as f:
            for line in f:
                if line.startswith("iteration="):
                    iteration = int(line.split("=")[1].strip())
                elif line.startswith("episode="):
                    episode = int(line.split("=")[1].strip())
    except Exception as e:
        print(f"读取进度配置文件失败: {e}")
    return iteration, episode

def _bao_db():
    conn = sqlite3.connect("bao.db")
    c = conn.cursor()
    c.execute("""
CREATE TABLE IF NOT EXISTS experience (
    id INTEGER PRIMARY KEY,
    pg_pid INTEGER,
    plan TEXT, 
    reward REAL,
    iteration INTEGER,
    episode INTEGER
)""")
    c.execute("""
CREATE TABLE IF NOT EXISTS experimental_query (
    id INTEGER PRIMARY KEY, 
    query TEXT UNIQUE,
    iteration INTEGER,
    episode INTEGER
)""")
    c.execute("""
CREATE TABLE IF NOT EXISTS experience_for_experimental (
    experience_id INTEGER,
    experimental_id INTEGER,
    arm_idx INTEGER,
    iteration INTEGER,
    episode INTEGER,
    FOREIGN KEY (experience_id) REFERENCES experience(id),
    FOREIGN KEY (experimental_id) REFERENCES experimental_query(id),
    PRIMARY KEY (experience_id, experimental_id, arm_idx)
)""")
    conn.commit()
    return conn


def record_reward(plan, reward, pid):
    iteration, episode = read_progress()
    with _bao_db() as conn:
        c = conn.cursor()

        # 检查当前行数
        c.execute("SELECT COUNT(*) FROM experience")
        row_count = c.fetchone()[0]

        if row_count < ROW_LIMIT:
            # 如果行数少于500，正常插入
            c.execute("INSERT INTO experience (plan, reward, pg_pid, iteration, episode) VALUES (?, ?, ?, ?, ?)",
                      (json.dumps(plan), reward, pid, iteration, episode))
        else:
            # 如果行数达到500，替换最旧的记录
            # 找出最旧记录的ID
            c.execute("SELECT id FROM experience ORDER BY id ASC LIMIT 1")
            oldest_id = c.fetchone()[0]

            # 更新最旧的记录
            c.execute("UPDATE experience SET plan = ?, reward = ?, pg_pid = ?, iteration = ?, episode = ? WHERE id = ?",
                      (json.dumps(plan), reward, pid, iteration, episode, oldest_id))

        conn.commit()

    print("Logged reward of", reward)


def last_reward_from_pid(pid):
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM experience WHERE pg_pid = ? ORDER BY id DESC LIMIT 1",
                  (pid,))
        res = c.fetchall()
        if not res:
            return None
        return res[0][0]

def experience():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT plan, reward FROM experience")
        return c.fetchall()

def experience_episode(iteration, episode):
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT plan, reward FROM experience WHERE iteration = ? AND episode = ?",
                  (iteration, episode))
        return c.fetchall()

def experience_iteration(iteration):
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT plan, reward FROM experience WHERE iteration = ?",
                  (iteration,))
             
        return c.fetchall()
    
def experiment_experience():
    all_experiment_experience = []
    for res in experiment_results():
        all_experiment_experience.extend(
            [(x["plan"], x["reward"]) for x in res]
        )
    return all_experiment_experience
    
def experience_size():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM experience")
        return c.fetchone()[0]

def clear_experience():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM experience")
        conn.commit()


def record_experimental_query(sql):
    with _bao_db() as conn:
        c = conn.cursor()

        # 检查当前行数
        c.execute("SELECT COUNT(*) FROM experimental_query")
        row_count = c.fetchone()[0]

        if row_count < ROW_LIMIT:
            # 如果行数少于500，正常插入
            try:
                c.execute("INSERT INTO experimental_query (query) VALUES (?)", (sql,))
                conn.commit()
                print("Added new test query.")
            except sqlite3.IntegrityError as e:
                raise BaoException("Could not add experimental query. "
                                   + "Was it already added?") from e
        else:
            # 替换最旧的记录
            c.execute("SELECT id FROM experimental_query ORDER BY id ASC LIMIT 1")
            oldest_id = c.fetchone()[0]
            c.execute("UPDATE experimental_query SET query = ? WHERE id = ?", (sql, oldest_id))
            conn.commit()
            print("Replaced oldest test query.")

def num_experimental_queries():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("SELECT count(*) FROM experimental_query")
        return c.fetchall()[0][0]
# FIXME Qihan: never used
def unexecuted_experiments():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("CREATE TEMP TABLE arms (arm_idx INTEGER)")
        # now we have 49 arms
        c.execute("INSERT INTO arms (arm_idx) VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),\
                  (11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),\
                  (27),(28),(29),(30),(31),(32),(33),(34),(35),(36),(37),(38),(39),(40),(41),(42),\
                  (43),(44),(45),(46),(47), (48)")

        c.execute("""
SELECT eq.id, eq.query, arms.arm_idx 
FROM experimental_query eq, arms
LEFT OUTER JOIN experience_for_experimental efe 
     ON eq.id = efe.experimental_id AND arms.arm_idx = efe.arm_idx
WHERE efe.experience_id IS NULL
""")
        return [{"id": x[0], "query": x[1], "arm": x[2]}
                for x in c.fetchall()]

def experiment_results():
    with _bao_db() as conn:
        c = conn.cursor()
        c.execute("""
SELECT eq.id, e.reward, e.plan, efe.arm_idx
FROM experimental_query eq, 
     experience_for_experimental efe, 
     experience e 
WHERE eq.id = efe.experimental_id AND e.id = efe.experience_id
ORDER BY eq.id, efe.arm_idx;
""")
        for eq_id, grp in itertools.groupby(c, key=lambda x: x[0]):
            yield ({"reward": x[1], "plan": x[2], "arm": x[3]} for x in grp)
        
def record_experiment(experimental_id, experience_id, arm_idx):
    with _bao_db() as conn:
        c = conn.cursor()

        # 检查当前行数
        c.execute("SELECT COUNT(*) FROM experience_for_experimental")
        row_count = c.fetchone()[0]

        if row_count < ROW_LIMIT:
            # 如果行数少于500，正常插入
            c.execute("""
            INSERT INTO experience_for_experimental (experience_id, experimental_id, arm_idx)
            VALUES (?, ?, ?)""", (experience_id, experimental_id, arm_idx))
        else:
            # 替换最旧的记录
            c.execute("SELECT experience_id, experimental_id, arm_idx FROM experience_for_experimental ORDER BY experience_id, experimental_id, arm_idx ASC LIMIT 1")
            oldest_ids = c.fetchone()
            c.execute("""
            UPDATE experience_for_experimental 
            SET experience_id = ?, experimental_id = ?, arm_idx = ?
            WHERE experience_id = ? AND experimental_id = ? AND arm_idx = ?""",
                      (experience_id, experimental_id, arm_idx, oldest_ids[0], oldest_ids[1], oldest_ids[2]))
        
        conn.commit()
        print("Recorded experiment.")