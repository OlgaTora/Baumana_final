import pandas as pd
from airflow.utils.dates import days_ago
from airflow.decorators import dag, task

from dags.config.config import db, input_path, archive_path
from dags.etl.cleaner import Cleaner
import os
from dags.etl.connection import Connection
from dags.etl.transform import Transform
from dags.models.linear_model import LinearModel
from models.logisticreg_model import LogisticRegressionModel
from dags.etl.splitter import Splitter
from dags.etl.scaler import Scaler
from dags.etl.preprocessing import Preprocessing
from dags.etl.extractdata import ExtractData
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

d = dict(
    AIRFLOW_UID="$(id -u)",
    AWS_ACCESS_KEY_ID="admin",
    AWS_SECRET_ACCESS_KEY="sample_key",
    AWS_REGION="us - east - 1",
    AWS_BUCKET_NAME="mlflow",
    MYSQL_DATABASE="mlflow",
    MYSQL_USER="mlflow_user",
    MYSQL_PASSWORD="mlflow_password",
    MYSQL_ROOT_PASSWORD="toor",
    MLFLOW_TRACKING_URI='http://mlflow:5001',
    MLFLOW_EXPERIMENT_NAME="MyExp",
    MLFLOW_S3_ENDPOINT_URL='http://s3:9000'
)
for i in d:
    os.environ[i] = d[i]


@dag(
    schedule=None,
    start_date=days_ago(0),
    catchup=False,
    tags=['model'],
)
def get_model(debug=True):

    @task()
    def extract_data_from_files() -> list:
        connect = Connection(db).create_connection()
        extract = ExtractData(input_path, archive_path, connect)
        tables_list = extract.create_stg_tables()
        return tables_list

    @task()
    def create_dwh(tables_list: list) -> list:
        dwh_tables_list = []
        connect = Connection(db).create_connection()
        for table in tables_list:
            data = pd.read_sql_query(f"""Select * from {table}""", con=connect)
            transformer = Transform(data)
            transform_data = transformer.transform()
            transform_data.to_sql(f'DWH_{table[4:]}', con=connect, if_exists='replace', index=False)
            # CHAMGE APPEND
            dwh_tables_list.append(f'DWH_{table[4:]}')
        return dwh_tables_list

    @task()
    def create_data_mart(tables_list: list) -> str:
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
        connect = Connection(db).create_connection()
        data = pd.read_sql_query(f"""Select * from {mart_table}""", con=connect)
        process = Preprocessing(data)
        process_data = process.process_data()
        process_data.to_sql('cleaned_data', con=connect, if_exists='replace', index=False)
        return 'cleaned_data'

    @task(multiple_outputs=True)
    def train_test_split(mart_table: str) -> dict:
        connect = Connection(db).create_connection()
        data = pd.read_sql_query(f"""Select * from {mart_table}""", con=connect)
        splitter = Splitter()
        train_data, test_data = splitter.split_test_train(data)
        train_data.to_sql('train_data', con=connect, if_exists='replace', index=False)
        test_data.to_sql('test_data', con=connect, if_exists='replace', index=False)
        return {'test_data': 'test_data', 'train_data': 'train_data'}

    @task(multiple_outputs=True)
    def model_materialize(model_name, test_data):
        logged_model = f'models:/{model_name}/None'
        conn = Connection(db).create_connection()
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
        predict = loaded_model.predict(test.drop(columns=['target']))
        test['predict'] = predict
        test[['predict']].to_sql('model_predict', con=conn, if_exists='replace')

    @task()
    def model_fit(train_data, test_data):
        connect = Connection(db).create_connection()
        train = pd.read_sql_query(f"""Select * from {train_data}""", con=connect)

        # with mlflow.start_run() as run:
        #     train = pd.read_sql_query(f"""Select * from {train_data}""", con=conn)
        #     lr = LogisticRegressionModel()
        #     lr.fit(train.drop(columns=['target']), train['target'])
        #     test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
        #     preds = lr.prediction(test.drop(columns=['target']))
        #     signature = infer_signature(train.drop(columns=['target']), preds)
        #     mlflow.sklearn.log_model(lr, "lr_model", signature=signature, registered_model_name='LinearModel')
        # signature = infer_signature(iris_train, clf.predict(iris_train))
        # mlflow.set_tracking_uri('http://mlflow')
        # experiment_id = mlflow.create_experiment("training experiment_1")
        # mlflow.set_experiment('experiment')

        mmodel = LinearModel()
        with mlflow.start_run():
            mmodel.fit(train.drop(columns=['target']), train['target'])
            test = pd.read_sql_query(f"""Select * from {test_data}""", con=connect)
            preds = mmodel.model.predict(train.drop(columns=['target']))
            signature = infer_signature(train.drop(columns=['target']), preds)
            mlflow.sklearn.log_model(mmodel.model, "model", signature=signature, registered_model_name='LinearModel')
        return 'LinearModel'

    @task()
    def get_metrics(model_name, test_data):
        logged_model = f'models:/{model_name}/None'
        conn = Connection(db).create_connection()
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(logged_model)
        model = LogisticRegressionModel()
        model.model = loaded_model
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
        metrics = model.get_metrics(test.drop(columns=['target']), test['target'])

        for j in metrics:
            mlflow.log_metric(j, metrics[j])

    # @task(multiple_outputs=False)
    # def fit_predict(train_data: str, test_data: str, debug_mode=True) -> dict:
    #     connect = Connection(db).create_connection()
    #     train = pd.read_sql_query(f"""Select * from {train_data}""", con=connect)
    #     test = pd.read_sql_query(f"""Select * from {test_data}""", con=connect)
    #     splitter = Splitter()
    #     X_train, y_train = splitter.split_x_y(train)
    #     X_test, y_test = splitter.split_x_y(test)
    #     scaler = Scaler()
    #     X_train = scaler.scaling(X_train)
    #     X_test = scaler.scaling(X_test)
    #     model = LinearModel()
    #     model.fit(X_train, y_train)
    #     if debug_mode:
    #         metrics = model.get_metrics(X_test, y_test)
    #         return metrics
    #     else:
    #         y_pred = model.predict(X_test)
    #         test['predict'] = y_pred
    #         test.to_sql('model_predict', con=connect, if_exists='replace', index=False)
    #         return {'y_pred': 'y_pred'}

    data = extract_data_from_files()
    dwh_tables_list = create_dwh(data)
    mart_table = create_data_mart(dwh_tables_list)
    prep_data = preprocessing(mart_table)
    train_test = train_test_split(prep_data)
    model = model_fit(train_test['train_data'], train_test['test_data'])
    # result = fit_predict(train_test['train_data'], train_test['test_data'], debug_mode=debug)
    model_mat = model_materialize(model, train_test['test_data'])


# get_metrics(model, train_test['test_data'])

#
# FROM python:3.9.13-slim
#
# WORKDIR /app
#
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir 'mlflow==1.26.1'
#
# CMD mlflow server \
#     --backend-store-uri sqlite:////app/mlruns.db \
#     --default-artifact-root $ARTIFACT_ROOT \
#     --host 0.0.0.0

prod = get_model()
