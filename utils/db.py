import pandas as pd
import sqlite3
from pandas.core.frame import DataFrame
import os
import duckdb
import numpy as np

duckdb_type = {
    'object': 'TEXT',
    'int64': 'INTEGER',
    'float64': 'DOUBLE',
    "bool": "BOOLEAN",
    'datetime64': 'TIMESTAMP',
    'datetime64[ns]': 'TIMESTAMP',
}

class dbconn(object):
    def __init__(self):
        raise NotImplementedError()
    
    def query(self):
        raise NotImplementedError()
    
    def insert(self):
        raise NotImplementedError()
    
    def list_databases(self):
        raise NotImplementedError()

    def list_tables(self):
        raise NotImplementedError()

    def drop_database(self):
        raise NotImplementedError()
    
    def drop_table(self):
        raise NotImplementedError()

class sqlite(dbconn):
    def __init__(self, database):
        self.database = f"data/sqlite/{database}.db"
        pass

    def query(self, sql_query: str) -> DataFrame:
        try:
            conn = sqlite3.connect(self.database)
            df = pd.read_sql(sql_query, conn)
            return df
        finally:
            conn.close()        

    def insert(self, table, data: DataFrame):
        try:
            conn = sqlite3.connect(self.database)
            data.to_sql(table, conn, if_exists='append', index=False)
        finally:
            conn.close()

    def list_databases(self) -> DataFrame:    
        return self.query("PRAGMA database_list")

    def list_tables(self) -> DataFrame:
        return self.query("SELECT name FROM sqlite_master WHERE type='table'")    
    
    def drop_database(self):        
        if os.path.exists(self.database):
            os.remove(self.database)

    def drop_table(self, table: str):
        self.query("DROP TABLE {table}")

class duck(dbconn):

    def __init__(self, database):
        self.database = f"data/duckdb/{database}.db"

    def query(self, sql_query: str) -> DataFrame:
        try:
            conn = duckdb.connect(database=self.database, read_only=False)
            df = conn.execute(sql_query).fetchdf()
            return df
        finally:
            conn.close()                

    def insert(self, table, data: DataFrame, append_new_column=False):
        try:
            conn = duckdb.connect(database=self.database, read_only=False)
            data.to_sql(table, conn, if_exists='append', index=False)
        except Exception as e:
            '''
                Additional logic to alter table to add new columns in case insert dataframe has different schema
            '''
            if append_new_column:
                curr_cols = self.query(f"SELECT * FROM {table} LIMIT 1").columns
                new_cols = list(np.setdiff1d(data.columns, curr_cols))
                if len(new_cols):
                    for col in new_cols:
                        alter_sql = f"ALTER TABLE {table} ADD COLUMN {col} {duckdb_type[str(data.dtypes[col])]}" 
                        print(alter_sql)
                        self.query(alter_sql)                                     

                    try:
                        data.to_sql(table, conn, if_exists='append', index=False)                        
                    except Exception as e:
                        raise Exception(e)
                else:
                    raise Exception(e)
            else:
                raise Exception(e)
        finally:
            conn.close()

    def list_databases(self) -> DataFrame:    
        files = [f for f in os.listdir("data/duckdb/") if os.path.isfile(os.path.join("data/duckdb/", f))]
        files = pd.DataFrame({'database': files})
        return files

    def list_tables(self) -> DataFrame:
        return self.query("SHOW TABLES")    
    
    def drop_database(self):        
        if os.path.exists(self.database):
            os.remove(self.database)

    def drop_table(self, table: str):
        self.query(f"DROP TABLE {table}")



            

        
    
    