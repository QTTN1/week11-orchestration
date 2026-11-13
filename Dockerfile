# Airflow image + the extra libraries our tasks need.
FROM apache/airflow:2.10.5-python3.11

COPY requirements.txt /requirements.txt
# Pinning apache-airflow in the same install stops pip from upgrading/downgrading Airflow itself.
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /requirements.txt
