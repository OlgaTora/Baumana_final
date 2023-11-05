import requests
from bs4 import BeautifulSoup
import pandas as pd
from dags.logger import logger

"""https://www.velostrana.ru/veloaksessuary/"""


class Parser:

    def __init__(self, category_name):
        self.category_name = category_name

    def get_data(self, page_number):
        response = requests.get(f'https://www.velostrana.ru/veloaksessuary/{self.category_name}/{page_number}.html')
        soup = BeautifulSoup(response.text)
        all_items = soup.find_all('div', {"class": 'product-card'})
        logger.info('Get data from site')
        if len(all_items):
            return all_items
        else:
            return None

    @staticmethod
    def create_dataframe(items):
        d = {'item_name': [],
             'price': []}

        for item in items:
            name = item.find_all('div', {'class': 'product-card__model'})
            if len(name):
                d['item_name'].append(name[0].text)
            price = item.find_all('div', {'class': 'product-card__price-new'})
            if len(price):
                price = price[0].text.replace(' ', '').replace('руб', '').replace('₽', '')
                d['price'].append(price)
            else:
                d['price'].append("NaN")
        logger.info('Create dataframe from 1 page')
        return pd.DataFrame(d)

    def take_all_files(self):
        buf = []
        page_number = 1
        while True:
            data = self.get_data(page_number)
            print(f'{page_number} page_number')
            if data is None:
                break
            df = self.create_dataframe(data)
            buf.append(df)
            page_number += 1
        full_data = pd.concat(buf)
        logger.info('Create dataframe from all pages in category')
        return full_data
