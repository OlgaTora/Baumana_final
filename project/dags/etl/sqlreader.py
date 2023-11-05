import sqlite3

from dags.logger import logger


class SQLReader:
    def __init__(self, connection):
        self.connection = connection

    @staticmethod
    def open_sql_file(file_path: str) -> str:
        """Function for open sql script"""
        with open(file_path, 'r') as sql_file:
            return sql_file.read()

    def execute_sql_script(self, sql_path: str, file: bool = True) -> None:
        """Function for execute sql script to table"""
        connect = self.connection.create_connection()
        try:
            cur = connect.cursor()
            if file:
                sql_script = self.open_sql_file(sql_path)
            else:
                sql_script = sql_path
            cur.executescript(sql_script)
            connect.commit()
        except ConnectionError as conn_error:
            logger.info(conn_error)
        except FileNotFoundError as file_error:
            logger.info(file_error)
        except sqlite3.Error as sql_error:
            logger.info(sql_error)
            if connect:
                connect.rollback()
        finally:
            if connect:
                connect.close()
