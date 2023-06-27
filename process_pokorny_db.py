import re
import sqlite3

import mysql.connector
import pandas as pd
from tqdm import tqdm


# Function to extract schema from the database dump file
def extract_from_dump(dump_file_path):
    # Connect to the MySQL database
    cnx = mysql.connector.connect(
        host='localhost',
        user='root',
        password='velociped',
        database='IELEX',
        charset='utf8'
    )

    # Create a cursor object to execute SQL queries
    cursor = cnx.cursor()

    # Get the list of table names in the database
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]

    # Dictionary to store the DataFrames
    dfs = {}

    # Extract data from each table
    for table in tables:
        # Execute a SELECT query for the table
        query = f"SELECT * FROM `{table}`"
        cursor.execute(query)

        # Fetch all the rows as a list of tuples
        rows = cursor.fetchall()

        # Get the column names
        columns = [desc[0] for desc in cursor.description]

        # Create a DataFrame from the rows and columns
        df = pd.DataFrame(rows, columns=columns)

        # Store the DataFrame in the dictionary with the table name as the key
        dfs[table] = df

    # Close the cursor and connection
    cursor.close()
    cnx.close()

    return dfs


def main():
    sql_dump_file = 'data_pokorny/lrc.sql'
    dfs = extract_from_dump(sql_dump_file)
    for name, df in dfs.items():
        df.to_pickle(f"data_pokorny/table_dumps/{name}.df")
    breakpoint()


if __name__ == '__main__':
    main()
    pass
