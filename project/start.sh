mkdir -p ./dags ./logs dags/archive
echo "AIRFLOW_UID=$(id -u)
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=sample_key
AWS_REGION=us-east-1
AWS_BUCKET_NAME=mlflow
MYSQL_DATABASE=mlflow
MYSQL_USER=mlflow_user
MYSQL_PASSWORD=mlflow_password
MYSQL_ROOT_PASSWORD=toor
MLFLOW_S3_ENDPOINT_URL=http://s3:9000
MLFLOW_TRACKING_URI=http://mlflow:5001"> .env
docker-compose up
