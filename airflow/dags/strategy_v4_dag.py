from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from strategy_v4.Data.run import run as run_data
from strategy_v4.Model.run import run as run_model
from strategy_v4.Evaluate.run import run as run_eval
from strategy_v4.config import MODEL_LAYER


with DAG(
    "Return_Prediction",    
    default_args={
        "depends_on_past": False,        
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,        
        "retry_delay": timedelta(seconds=5),
    },
    description="Return_Prediction",
    schedule_interval=None,        
) as dag:
        
    with TaskGroup("data_layer", tooltip="Data Layer"):    
        data_layer = PythonOperator(
            task_id="data",
            python_callable=run_data,
            op_args=[],
            dag=dag
        )

    with TaskGroup("model_layer", tooltip="Model Layer"):
        model_layer = [
            PythonOperator(
                task_id=f"model_{model.lower()}",
                python_callable=run_model,
                op_args=[model],
                dag=dag
            ) 
            for model in MODEL_LAYER.models
        ]

    with TaskGroup("eval_layer", tooltip="Evaluation Layer"):
        eval_layer = PythonOperator(
            task_id="eval",
            python_callable=run_eval,
            dag=dag
        )

    data_layer >> model_layer >> eval_layer



    