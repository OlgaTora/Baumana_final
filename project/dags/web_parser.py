from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from config.config import db
from dags.etl.connection import Connection
from dags.etl.sqlreader import SQLReader
from dags.parser.parser import Parser
from dags.sql_scripts.table_creator import TableCreator


@dag(
    schedule=None,
    start_date=days_ago(0),
    catchup=False,
    tags=['parser'],
)
def parse():

    @task()
    def extract_web_data(category_name):
        connect = Connection(db).create_connection()
        parser = Parser(category_name)
        data = parser.take_all_files()
        data['category'] = category_name
        table_name = category_name.replace('-', '_')
        table_name = f'STG_{table_name}'
        data.to_sql(table_name, con=connect, if_exists='replace')
        return table_name

    @task()
    def save_parse_data2dwh(list_tables):
        connect = Connection(db)
        reader = SQLReader(connect)
        table_name = 'DWH_parser'
        columns = ['item_name', 'price']
        table_creator = TableCreator(table_name, columns, list_tables)
        table = table_creator.create_table()
        reader.execute_sql_script(table, file=False)
        return 'DWH_parser'

    category_list = ['velosipednie-sumki', 'fonari', 'zamki', 'krilya', 'flagi', 'bagajniki']
    extract_data = extract_web_data.expand(category_name=category_list)
    dwh_tables = save_parse_data2dwh(extract_data)

prod = parse()
