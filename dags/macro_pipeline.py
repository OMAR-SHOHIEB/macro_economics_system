from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "omar",
    "retries": 1,
}

PYTHON_ENV = "export PYTHONPATH=/usr/local/airflow/include:$PYTHONPATH && "

with DAG(
    dag_id="macro_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
) as dag:


    world_bank_task = BashOperator(
        task_id="world_bank_scraping",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/scraping/world_bank.py"
    )

    imf_task = BashOperator(
        task_id="imf_scraping",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/scraping/imf.py"
    )


    clean_task = BashOperator(
        task_id="clean_data",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/processing/clean.py"
    )

    feature_extraction_task = BashOperator(
        task_id="feature_extraction",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/processing/feature_extraction.py"
    )

    feature_selection_task = BashOperator(
        task_id="feature_selection",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/processing/feature_selection.py"
    )

    prepare_data_task = BashOperator(
        task_id="prepare_data",
        bash_command=PYTHON_ENV + "python -u /usr/local/airflow/include/pipeline/processing/prepare_data.py"
    )


    [world_bank_task ,imf_task] >> clean_task

    clean_task >> feature_extraction_task

    feature_extraction_task >> feature_selection_task

    feature_selection_task >> prepare_data_task