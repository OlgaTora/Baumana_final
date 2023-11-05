FROM apache/airflow:2.7.1

COPY ./.env ./
COPY ./docker/airflow/requirements.txt ./
RUN pip install pip==23.1.2
RUN pip install --no-cache-dir -r requirements.txt
COPY ./dags ./dags

ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow/dags"

