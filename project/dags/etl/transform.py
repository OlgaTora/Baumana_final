import pandas as pd
from dags.logger import logger


class Transform:

    def __init__(self, data):
        self.data = data

    def transform(self):
        self.data = self.transform_dates()
        logger.info('Data features transform to data format')
        return self.data

    def transform_dates(self):
        """Function for transform dates to format datetime and fill N/A"""
        if {'transaction_date', 'product_first_sold_date'}.issubset(self.data.columns):
            most_freq = self.data['product_first_sold_date'].mode()
            self.data['product_first_sold_date'].fillna(most_freq, inplace=True)
            self.data['product_first_sold_date'] = self.data['product_first_sold_date'].astype(float)
            self.data['product_first_sold_date'] = (
                    pd.to_datetime("1900-01-01") + pd.to_timedelta(self.data['product_first_sold_date'], unit="D"))
            self.data['transaction_date'] = pd.to_datetime(self.data['transaction_date'])
        if 'DOB' in self.data.columns:
            self.data['DOB'] = pd.to_datetime(self.data['DOB'], format='mixed')
        return self.data
