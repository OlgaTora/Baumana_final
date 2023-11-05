from sklearn.model_selection import train_test_split
from dags.logger import logger


class Splitter:

    def split_test_train(self, data) -> tuple:
        train_data, test_data = train_test_split(data, random_state=42, test_size=0.2)
        logger.info('Data split for test and train')
        return train_data, test_data

    @staticmethod
    def split_x_y(data) -> tuple:
        X = data.drop(columns=['target'], axis=1)
        y = data['target']
        logger.info('Data split for X and y')
        return X, y
