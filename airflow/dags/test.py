from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    'owner': 'admin',
    'start_date': dt.datetime(2023, 12, 1),
    'retries': 1,
    'retry_delays': dt.timedelta(minutes=1),
    'depends_on_past': False,
    'provide_context': True
}

with DAG(
    dag_id='test',
    default_args=args,
    schedule_interval=None,
    tags=['football', 'score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/datasets && gdown 1j7HA_b0U8GZuHoPoVgDRtPwQFg9cALWX", 
                            dag=dag)
    extract_data = BashOperator(task_id='extract_data',
                            bash_command="mkdir raw && unzip -o -q /home/xypma/lab3/MLOps_lab3/datasets/data.zip -d /home/xypma/lab3/MLOps_lab3/datasets/raw/", 
                            dag=dag)
    merge_df = BashOperator(task_id='merge_df',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/merge_df.py", 
                            dag=dag)
    clear_features = BashOperator(task_id='clear_features',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/clear_features.py", 
                            dag=dag)
    modify_df = BashOperator(task_id='modify_df',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/modify_df.py", 
                            dag=dag)
    train_test_split = BashOperator(task_id='train_test_split',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/train_test_split.py", 
                            dag=dag)
    model_learn = BashOperator(task_id='model_learn',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/model_learn.py", 
                            dag=dag)
    evaluate = BashOperator(task_id='evaluate',
                            bash_command="cd /home/xypma/lab3/MLOps_lab3/ && python /home/xypma/lab3/MLOps_lab3/scripts/eval.py", 
                            dag=dag)

    get_data >> extract_data >> merge_df >> clear_features >> modify_df >> train_test_split >> model_learn >> evaluate
