import os
import glob
import time
import psycopg2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, body, to_email):
    """
    Send an email using a QQ email account.
    """
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
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def execute_sql_files(directory, connection_str):
    """
    Execute all SQL files in the given directory using the provided PostgreSQL connection string.
    Returns the file name with the longest execution time and its execution time in seconds.
    """
    # Connect to the database
    conn = psycopg2.connect(connection_str)
    cur = conn.cursor()
    
    max_time = 0
    max_file = None
    
    # Retrieve all .sql files in the directory
    sql_files = glob.glob(os.path.join(directory, "*.sql"))
    for file in sql_files:
        with open(file, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        start_time = time.time()
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"Error executing file {file}: {e}")
            conn.rollback()
            continue
        
        exec_time = time.time() - start_time
        print(f"File: {file} executed in {exec_time:.4f} seconds.")
        
        if exec_time > max_time:
            max_time = exec_time
            max_file = file
            
    cur.close()
    conn.close()
    return max_file, max_time

def run_experiment(name, connection_str, directories):
    """
    Run the SQL execution experiment for a given set of directories and a PostgreSQL connection string.
    Returns a summary string of the longest execution times for each directory.
    """
    summary_lines = [f"Results for {name} experiment:"]
    for directory in directories:
        longest_file, longest_time = execute_sql_files(directory, connection_str)
        if longest_file:
            summary_lines.append(
                f"Directory: {directory} - Longest SQL file: {longest_file}, Execution time: {longest_time:.4f} seconds."
            )
        else:
            summary_lines.append(f"Directory: {directory} - No SQL file executed.")
    summary = "\n".join(summary_lines)
    print(summary)
    return summary

def main():
    # Define experiments with their corresponding connection strings and SQL directories
    experiments = [
        {
            "name": "Stack",
            "connection_str": "dbname=soload user=qihan host=localhost",
            "directories": [
                "/mydata/LIMAOLifeLongRLDB/so_assorted",
                "/mydata/LIMAOLifeLongRLDB/so_assorted_2"
            ]
        },
        {
            "name": "imdb",
            "connection_str": "dbname=imdbload user=qihan host=localhost",
            "directories": [
                "/mydata/LIMAOLifeLongRLDB/imdb_assorted_3",
                "/mydata/LIMAOLifeLongRLDB/imdb_assorted_4"
            ]
        },
        {
            "name": "tpch",
            "connection_str": "dbname=tpch10load user=qihan host=localhost",
            "directories": [
                "/mydata/LIMAOLifeLongRLDB/tpch_assorted",
                "/mydata/LIMAOLifeLongRLDB/tpch_assorted_2",
                "/mydata/LIMAOLifeLongRLDB/tpch_assorted_3"
            ]
        }
    ]
    
    recipient_email = "2453939195@qq.com"
    
    # Process each experiment and send an email summary
    for experiment in experiments:
        summary = run_experiment(
            experiment["name"],
            experiment["connection_str"],
            experiment["directories"]
        )
        send_email("Timeout Experiment", summary, recipient_email)

if __name__ == "__main__":
    main()
