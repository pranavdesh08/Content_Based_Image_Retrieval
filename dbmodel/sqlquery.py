import os
import sqlite3
import pandas as pd

df = pd.read_csv('C:/Users/12014/Documents/Image retrieval/Trial_1/catlog_categories.csv',index_col=0)

# Create a database
conn = sqlite3.connect('example.db', check_same_thread=False)

# Add the data to our database
df.to_sql('data_table', conn, if_exists='replace', dtype={
    'index':'VARCHAR(256)',
    'picture_name':'VARCHAR(256)',
    'category':'VARCHAR(256)',
    'client_id':'VARCHAR(256)',
    'price':'VARCHAR(256)',
    'product_name':'VARCHAR(256)',
    'color': 'VARCHAR(256)',
})
    
conn.row_factory = sqlite3.Row

# Make a convenience function for running SQL queries
def sql_query(query):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return rows

def sql_edit_insert(query,var):
    cur = conn.cursor()
    print(query,var)
    cur.execute(query,var)
    conn.commit()
    cur.close()

def sql_delete(query,var):
    cur = conn.cursor()
    cur.execute(query,var)
    conn.commit()
    cur.close()

def sql_query2(query,var):
    cur = conn.cursor()
    cur.execute(query,var)
    rows = cur.fetchall()
    return rows