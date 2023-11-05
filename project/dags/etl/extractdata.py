import shutil

import pandas as pd
import os
from dags.config.constants import SHEETS
from dags.logger import logger


class ExtractData:

    def __init__(self, input_path, archive_path, connection):
        self.input_path = input_path
        self.connection = connection
        self.archive_path = archive_path

    @staticmethod
    def extract_data_sheet(filename: str, sheet_name: str) -> pd.DataFrame:
        """Function for read file sheet and resolve problem with name 'default"""
        data = pd.read_excel(filename, sheet_name)
        for column in list(data):
            if column == 'default':
                data = data.rename(columns={'default': 'default_'})
                data['default_'] = data['default_'].astype(str)
        logger.info(f'Extract data from sheet {sheet_name}')
        return data

    def load(self, data, table_name):
        data.to_sql(table_name, con=self.connection, if_exists='replace', index=False)

    def create_stg_tables(self) -> list:
        input_files = os.listdir(self.input_path)
        archive_files = os.listdir(self.archive_path)
        new_files = [x for x in input_files if x + '.dump' not in archive_files]
        tables_list = []
        for file in new_files:
            filename = f'{self.input_path}/{file}'
            for sheet in SHEETS:
                table_name = f'STG_{sheet}'
                data = self.extract_data_sheet(filename, sheet_name=sheet)
                self.load(data, table_name)
                tables_list.append(table_name)
            # remove '#' if work every day with new file
            #shutil.copy(f'{self.input_path}{file}', f'{self.archive_path}{file}.dump')
            #os.remove(f'{self.input_path}{file}')
        logger.info('Create STG tables')
        return tables_list
