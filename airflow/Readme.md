# Airflow

(Command + K then V to open markdown in VS Code)

# Start
``` 
airflow webserver
airflow scheduler
airflow celery worker
```

First time start the webserver might need to use <b>sudo</b>
```
sudo airflow webserver
```

Use this command to create a new user
```
airflow users  create --role Admin --username lok419 --email admin --firstname admin --lastname admin --password 1234
```

If it returns errors of some ports are in used, we can just kill those process via below commands
```
lsof -i tcp:<port>
kill <port>
```

# AirFlow CFG
For some reasons, MacOS can't use localExecutor, which i decided to use CeleryExecutor instead, which requires RabbitMQ. and somehow the fork doesn't work in machnie, so need to set execute_tasks_new_python_interpreter = True
```
dags_folder = /Users/lok419/Desktop/JupyterLab/Trading/airflow/dags
execute_tasks_new_python_interpreter = True
executor = CeleryExecutor

# postregs connection 
sql_alchemy_conn = postgresql+psycopg2://airflow_user:airflow_pass@localhost:5432/airflow_db url

[celery]
# postregs connection url
celery_result_backend = db+postgresql://airflow_user:airflow_pass@localhost:5432/airflow_db 

# rabbitMQ connection url
broker_url = amqp://guest:guest@localhost:5672// 
```

# Requirement
# Postregsql
```
brew install postgresql
brew services start postgresql
```    

### Initalize DB for airflow
```
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_pass';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;
GRANT ALL ON SCHEMA public TO airflow_user;
ALTER DATABASE airflow_db OWNER TO airflow_user;
```

# RabbitMQ
```
brew install rabbitmq
abbitmq-plugins enable rabbitmq_management
brew services start rabbitmq
```

can check status under http://localhost:15672/#/, user/pwd = guest

### Initalize RabbitMQ
```
rabbitmqctl add_user admin admin
rabbitmqctl set_user_tags admin administrator
rabbitmqctl set_permissions -p / admin "." "." ". "
```

# Celery

Celery is a python interface that wraps RabbitMQ 

```
airflow celery worker # run a workers
airflow celery flower # page to manage all workers
```

