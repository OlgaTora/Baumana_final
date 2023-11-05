#!/bin/bash

source .env

echo "Copy and paste below configuration into your ~/.bashrc file!"
echo ""
echo "# AIRFLOW CONFIG" 
echo "export AIRFLOW_UID=$AIRFLOW_UID"
echo "# END AIRFLOW CONFIG"
echo "# MLFLOW CONFIG" 
echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
echo "export AWS_REGION=$AWS_REGION"
echo "export MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL"
echo "export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "# END MLFLOW CONFIG"