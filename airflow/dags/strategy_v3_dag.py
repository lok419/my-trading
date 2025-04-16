from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.ExecuteSetup.StrategyFactory import StrategyFactory
from utils.logging import get_logger

def execute(*args, **kwargs):
    strategy_id = args[0]        
    logger = get_logger('DAG')

    if len(args) > 1:
        lookback_str = args[1]
        logger.info(f"Running strategy: {strategy_id} with lookback {lookback_str}")    
    else:
        lookback_str = None
        logger.info(f"Running strategy: {strategy_id}")    
        
    strategy = StrategyFactory().get(strategy_id)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id, reload=True)    
    strategy.run_once(lookback_str)        

with DAG(
    "Grid_BTCUSDT",    
    default_args={
        "depends_on_past": False,        
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,        
    },
    description="Grid Trading Strategy for BTCUSDT",
    schedule="@hourly",
    start_date=datetime(2024, 11, 13),
    catchup=False,
) as dag:

    # run
    run = PythonOperator(
        task_id="run_strategy",
        python_callable=execute,
        op_args=['BTCv1', '10 Days Ago'],
        dag=dag
    )

with DAG(
    "Grid_ETHUSDT",    
    default_args={
        "depends_on_past": False,        
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,        
    },
    description="Grid Trading Strategy for ETHUSDT",
    schedule="@hourly",
    start_date=datetime(2025, 4, 17),
    catchup=False,
) as dag:

    # run
    run = PythonOperator(
        task_id="run_strategy",
        python_callable=execute,
        op_args=['ETHv1', '10 Days Ago'],
        dag=dag
    )

with DAG(
    "Exchange_Arbitrage",    
    default_args={
        "depends_on_past": False,        
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,        
    },
    description="Binance Arbitrage Strategy for BTCUSDT",
    schedule_interval='*/1 * * * *',
    start_date=datetime(2025, 1, 14),
    catchup=False,
) as dag:
    
    PythonOperator(
        task_id="binance_arb",
        python_callable=execute,
        op_args=['binance_arb'],
        dag=dag
    )
    
    PythonOperator(
        task_id="binance_arb_zero_fees",
        python_callable=execute,
        op_args=['binance_arb_zero_fees'],
        dag=dag
    )  