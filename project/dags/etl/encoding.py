import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


class Encoding:

    def __init__(self, categorical_feautures):
        self.categorical_feautures = categorical_feautures

    def encoding_data(self, data):
        """Function for encoder categorical features"""
        data.gender = (data.gender.replace('F', 'Female').
                       replace('Femal', 'Female').
                       replace('M', 'Male').replace('U', 'Unknown'))

        for i in self.categorical_feautures:
            lb = LabelEncoder()
            data[i] = lb.fit_transform(data[i].astype(str))
        return data
