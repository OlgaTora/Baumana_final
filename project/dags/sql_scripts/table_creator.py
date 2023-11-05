class TableCreator:

    def __init__(self, table_name: str, columns='*', list_tables=None):
        self.table_name = table_name
        self.columns = columns
        self.list_tables = list_tables

    def union_equal_tables(self) -> str:
        """
        Function for join tables
        columns - selected columns, default=*
        """
        columns = ', '.join(self.columns)
        tables = [f"Select {columns} from {x}" for x in self.list_tables]
        return ("""

        union

        """.join(tables))

    def create_table(self):
        script = self.union_equal_tables()
        return f"""
        Create table if not EXISTS {self.table_name} as {script}
        """
