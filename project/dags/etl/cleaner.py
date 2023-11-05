from datetime import datetime
import pandas as pd

from dags.logger import logger


class Cleaner:
    def __init__(self, data):
        self.data = data

    def cleaning(self):
        self.data = self.transform_cat()
        self.data = self.create_age()
        self.data = self.drop_columns()
        return self.data

    def drop_columns(self):
        self.data = self.data.drop(columns=['DOB', 'default_', 'country'], axis=1)
        logger.info('Drop columns for data mart')
        return self.data

    def create_age(self):
        data_describe = self.data.describe(include='all')
        most_freq = data_describe['DOB']['top']
        self.data['DOB'].fillna(most_freq, inplace=True)
        self.data['DOB'] = pd.to_datetime(self.data['DOB'])
        self.data['age'] = (datetime.today() - self.data['DOB'])#.astype('<m8[Y]')
        logger.info('Create age column')
        return self.data

    def transform_cat(self):
        self.data.gender = (self.data.gender.replace('F', 'Female').
                            replace('Femal', 'Female').
                            replace('M', 'Male').replace('U', 'Unknown'))
        logger.info('Transform gender column')
        return self.data
