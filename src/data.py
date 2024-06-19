import os
import numpy as np
import pandas as pd
import mysql.connector as mariadb

class DBManager:
    def __init__(self, db_config: dict):
        self.host = db_config['host']
        self.port = db_config['port']
        self.user = db_config['user']
        self.password = db_config['password']
        self.database = db_config['database']
    
    def __connect(self):
        my_cnx = mariadb.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return my_cnx
    
    def execute(self, query: str, query_type: str):
        my_cnx = self.__connect()
        my_cursor = my_cnx.cursor()
        try:
            results = None
            my_cursor.execute(query)
            if (query_type == "SELECT") | (query_type == "SHOW"):
                cols = [d[0] for d in my_cursor.description]
                results = pd.DataFrame(my_cursor.fetchall(), columns=cols)
            my_cnx.commit()
        except Exception as err:
            print(err)
        finally:
            my_cursor.close()
            my_cnx.close()
        return results
    
    def concat_with_parentheses(self, lst: list, quote=''):
        prefix = f"({quote}"
        separator = f"{quote}, {quote}"
        suffix = f"{quote})"
        return prefix + separator.join(lst) + suffix
    
    def create_insert_query(self, tbl_nm: str, cols: list, quote: str='`'):
        vls = self.concat_with_parentheses(["%s"] * len(cols))
        cols = self.concat_with_parentheses(cols, quote=quote)
        query = f"INSERT IGNORE INTO {tbl_nm} {cols} VALUES {vls};"
        return query

    def insert_into_table(self, tbl_nm: str, data: pd.DataFrame) -> bool:
        if isinstance(data, np.ndarray):
            data = data.tolist()
        my_cnx = self.__connect()
        my_cursor = my_cnx.cursor()
        try:
            cols = list(data.columns)
            data = list(data.values)
            query = self.create_insert_query(tbl_nm, cols)
            if len(data) == 1:
                data = tuple(data[0])  # (V1, V2, ... , Vn)
                my_cursor.execute(query, data)
            else:
                data = tuple(
                    map(tuple, data)
                )  # ((V1, V2, ... , Vn), ... , (V1, V2, ... , Vn))
                my_cursor.executemany(query, data)
            my_cnx.commit()
            insert_success = True
        except Exception as err:
            insert_success = False
            print(err)
        finally:
            my_cursor.close()
            my_cnx.close()
        return insert_success


class DataManager:
    def __init__(self, conf):
        self.data_dir = "./data"
        self.data_filename = "casting.csv"
        self.db_config = conf.db_config
        self.dbm = DBManager(self.db_config)
        pass

    def read_data(self, data_dir: str=None, data_filename: str=None) -> pd.DataFrame:
        if data_dir is None:
            data_dir = self.data_dir
        if data_filename is None:
            data_filename = self.data_filename

        data_path = os.path.abspath(os.path.join(data_dir, data_filename))
        
        if data_filename.split(".")[-1] == "csv":
            data = pd.read_csv(data_path, encoding='euckr')
        else:
            raise ValueError("only csv file can be loaded now.")
        
        if data.columns[0] == "Unnamed: 0":
            data = data.set_index(data.columns[0])
            data.index.name = None
        
        return data

    def create_data_mart(self):
        data = self.read_data()
        return data
