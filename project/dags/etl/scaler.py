from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from dags.logger import logger


class Scaler:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def scaling(self, X):
        """Function for scaler features"""
        scaled_X = self.scaler.fit_transform(X)
        scaled_data = pd.DataFrame(scaled_X, columns=X.columns)
        logger.info('Data scaled with MinMaxScaler')
        return scaled_data
