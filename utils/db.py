import pandas as pd
import sqlite3
from pandas.core.frame import DataFrame
import os

class sqlite(object):
    def __init__(self, database):
        self.database = "data/sqlite/" + database
        pass

    def query(self, sql_query: str) -> DataFrame:
        conn = sqlite3.connect(self.database)
        df = pd.read_sql(sql_query, conn)
        conn.close()
        return df

    def insert(self, table, data: DataFrame):
        conn = sqlite3.connect(self.database)
        data.to_sql(table, conn, if_exists='append', index=False)
        conn.close()

    def list_tables(self) -> DataFrame:
        return self.query("SELECT name FROM sqlite_master WHERE type='table'")
    
    def list_databases(self) -> DataFrame:    
        return self.query("PRAGMA database_list")
    
    def drop_database(self):        
        if os.path.exists(self.database):
            os.remove(self.database)

    def drop_table(self, table: str):
        self.query("DROP TABLE {table}")

            

        
    
    