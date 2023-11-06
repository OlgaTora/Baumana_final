import pandas as pd
from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from imblearn.over_sampling import SMOTE

from config.config import db, input_path, archive_path
from etl.cleaner import Cleaner
import os
from etl.connection import Connection
from etl.transform import Transform
from models.linear_model import LinearModel
from etl.splitter import Splitter
from etl.scaler import Scaler
from etl.preprocessing import Preprocessing
from etl.extractdata import ExtractData
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

# d = dict(
#     AIRFLOW_UID="0",
#     AWS_ACCESS_KEY_ID="admin",
#     AWS_SECRET_ACCESS_KEY="sample_key",
#     AWS_REGION="us - east - 1",
#     AWS_BUCKET_NAME="mlflow",
#     MYSQL_DATABASE="mlflow",
#     MYSQL_USER="mlflow_user",
#     MYSQL_PASSWORD="mlflow_password",
#     MYSQL_ROOT_PASSWORD="toor",
#     MLFLOW_TRACKING_URI='http://mlflow:5001',
#     MLFLOW_EXPERIMENT_NAME="MyExp",
#     MLFLOW_S3_ENDPOINT_URL='http://s3:9000'
# )
# for i in d:
#     os.environ[i] = d[i]


@dag(
    schedule=None,
    start_date=days_ago(0),
    catchup=False,
    tags=['model'],
)
def get_model():
    @task()
    def extract_data_from_files() -> list:
        """Task for extract data from input files and save it in STG level"""
        connect = Connection(db).create_connection()
        extract = ExtractData(input_path, archive_path, connect)
        tables_list = extract.create_stg_tables()
        return tables_list

    @task()
    def create_dwh(tables_list: list) -> list:
        """Task for create DWH tables from STG level"""
        dwh_tables_list = []
        connect = Connection(db).create_connection()
        for table in tables_list:
            data = pd.read_sql_query(f"""Select * from {table}""", con=connect)
            transformer = Transform(data)
            transform_data = transformer.transform()
            transform_data.to_sql(f'DWH_{table[4:]}', con=connect, if_exists='replace', index=False)
            # change to "append" if work every day with new file
            dwh_tables_list.append(f'DWH_{table[4:]}')
        return dwh_tables_list

    @task()
    def create_data_mart(tables_list: list) -> str:
        """Task for create data mart table from DWH tables"""
        connect = Connection(db).create_connection()
        frames = []
        for table in tables_list:
            data = pd.read_sql_query(f"""Select * from {table}""", con=connect)
            frames.append(data)
        if len(frames) > 1:
            result = pd.merge(frames[0], pd.merge(frames[1], frames[2], on='customer_id'), on='customer_id')
        else:
            result = frames[0]
        cleaner = Cleaner(result)
        result = cleaner.cleaning()
        result.to_sql('DM_Sales', con=connect, if_exists='replace')
        return 'DM_Sales'

    @task()
    def preprocessing(mart_table: str) -> str:
        """Task for clean data from DWH tables"""
        connect = Connection(db).create_connection()
        data = pd.read_sql_query(f"""Select * from {mart_table}""", con=connect)
        process = Preprocessing(data)
        process_data = process.process_data()
        process_data.to_sql('cleaned_data', con=connect, if_exists='replace', index=False)
        return 'cleaned_data'

    @task(multiple_outputs=True)
    def train_test_split(mart_table: str) -> dict:
        """Task for split cleaned data"""
        connect = Connection(db).create_connection()
        data = pd.read_sql_query(f"""Select * from {mart_table}""", con=connect)
        splitter = Splitter()
        train_data, test_data = splitter.split_test_train(data)
        train_data.to_sql('train_data', con=connect, if_exists='replace', index=False)
        test_data.to_sql('test_data', con=connect, if_exists='replace', index=False)
        return {'test_data': 'test_data', 'train_data': 'train_data'}

    @task()
    def model_fit(train_data, test_data) -> str:
        """Task for fit model"""
        connect = Connection(db).create_connection()
        splitter = Splitter()
        scaler = Scaler()
        train = pd.read_sql_query(f"""Select * from {train_data}""", con=connect)
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=connect)
        X_train, y_train = splitter.split_x_y(train)
        smote = SMOTE(sampling_strategy='minority')
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test, y_test = splitter.split_x_y(test)
        X_train = scaler.scaling(X_train)
        X_test = scaler.scaling(X_test)
        lr = LinearModel()
        with mlflow.start_run():
            lr.fit(X_train, y_train)
            preds = lr.model.predict(X_train)
            signature = infer_signature(X_train, preds)
            mlflow.sklearn.log_model(lr.model, "model", signature=signature, registered_model_name='LinearModel')
        return 'LinearModel'

    @task(multiple_outputs=True)
    def model_materialize(model_name, test_data):
        """Task for materialize model"""
        logged_model = f'models:/{model_name}/None'
        connect = Connection(db).create_connection()
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=connect)
        predict = loaded_model.predict(test.drop(columns=['target']))
        test['predict'] = predict
        test[['predict']].to_sql('model_predict', con=connect, if_exists='replace')

    @task()
    def get_metrics(model_name, test_data) -> dict:
        """Task for get metrics from model"""
        logged_model = f'models:/{model_name}/None'
        connect = Connection(db).create_connection()
        loaded_model = mlflow.sklearn.load_model(logged_model)
        lr = LinearModel()
        lr.model = loaded_model
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=connect)
        metrics = lr.get_metrics(test.drop(columns=['target']), test['target'])
        for metric in metrics:
            mlflow.log_metric(metric, metrics[metric])
        return metrics

    data = extract_data_from_files()
    dwh_tables_list = create_dwh(data)
    mart_table = create_data_mart(dwh_tables_list)
    prep_data = preprocessing(mart_table)
    train_test = train_test_split(prep_data)
    model = model_fit(train_test['train_data'], train_test['test_data'])
    model_materialize(model, train_test['test_data'])
    get_metrics(model, train_test['test_data'])


prod = get_model()
