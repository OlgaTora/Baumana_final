import numpy as np
import pandas as pd
from dags.etl.encoding import Encoding
from dags.logger import logger


class Preprocessing:

    def __init__(self, data):
        self.data = data

    def process_data(self):
        data_copy = pd.DataFrame()
        data_copy['order_status'] = self.data['order_status']
        self.data = self.data.drop(columns=['order_status'], axis=1)
        self.data = self.update_categorical()
        self.data = self.update_numeric()
        self.data = self.drop_columns()
        self.data['target'] = data_copy['order_status']
        logger.info('Preprocessing complete')
        return self.data

    def drop_columns(self):
        """Function for drop columns"""
        self.data = self.data.drop(
            columns=['first_name',
                     'last_name',
                     'transaction_id',
                     'product_id',
                     'postcode',
                     'address',
                     'standard_cost',
                     'product_first_sold_date',
                     'transaction_date'],
            axis=1)
        logger.info('Drop columns for model')
        return self.data

    def update_numeric(self):
        """Function for fill N/A and transform numeric features to float type"""
        self.data = self.data.fillna(self.data.median(axis=0), axis=0)
        num_features = self.data.select_dtypes(include='number')
        for feature in num_features:
            self.data[feature] = self.data[feature].astype(float)
        logger.info('Numeric features transform')
        return self.data

    def update_categorical(self):
        """Function for fill N/A and encoder categorical features"""
        cat_features = self.data.select_dtypes(include='object')
        data_describe = self.data.describe(include=[object])
        for feature in cat_features:
            self.data[feature] = self.data[feature].fillna(data_describe[feature]['top'])
        encoder = Encoding(cat_features)
        self.data = encoder.encoding_data(self.data)
        logger.info('Categorical features transform')
        return self.data
