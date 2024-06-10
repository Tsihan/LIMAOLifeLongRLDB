import os
import re

def extract_aliases_from_sql(sql_text):
    # Regex to match "table_name AS alias" or "table_name as alias"
    pattern = re.compile(r'(\b\w+\b)\s+as\s+(\b\w+\b)', re.IGNORECASE)
    matches = pattern.findall(sql_text)
    return matches

def read_sql_files(directory):
    sql_files = [f for f in os.listdir(directory) if f.endswith('.sql')]
    all_aliases = []

    for sql_file in sql_files:
        with open(os.path.join(directory, sql_file), 'r', encoding='utf-8') as file:
            sql_text = file.read()
            aliases = extract_aliases_from_sql(sql_text)
            all_aliases.extend(aliases)

    return all_aliases

def main():
    current_directory1 = "/home/qihan/balsaLifeLongRLDB/queries/so_assorted_small"
    current_directory2 = "/home/qihan/balsaLifeLongRLDB/queries/so_assorted_small_2"

    if not os.path.exists(current_directory1):
        print(f"Directory {current_directory1} does not exist.")
        return
    if not os.path.exists(current_directory2):
        print(f"Directory {current_directory2} does not exist.")
        return

    aliases1 = read_sql_files(current_directory1)
    aliases2 = read_sql_files(current_directory2)
    
    aliases1 = set(aliases1)
    aliases2 = set(aliases2)
    
    print("Extracted aliases from directory 1:")
    for table, alias in aliases1:
        print(f"{table} AS {alias}")

    print("\nExtracted aliases from directory 2:")
    for table, alias in aliases2:
        print(f"{table} AS {alias}")

    print("\nAliases in directory 1 but not in directory 2:")
    for table, alias in (aliases1 - aliases2):
        print(f"{table} AS {alias}")

    print("\nAliases in directory 2 but not in directory 1:")
    for table, alias in (aliases2 - aliases1):
        print(f"{table} AS {alias}")

    # Compare the two sets for equality
    if aliases1 == aliases2:
        print("\nThe aliases in both directories are identical.")
    else:
        print("\nThe aliases in both directories are not identical.")

if __name__ == "__main__":
    main()
