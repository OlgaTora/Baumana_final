import sqlite3


class Connection:

    def __init__(self, conn_path: str):
        self.conn_path = conn_path

    def create_connection(self):
        return sqlite3.connect(self.conn_path)
