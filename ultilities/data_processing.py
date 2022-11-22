from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pandas as pd
import string

from google.cloud import bigquery
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'config/application_default_credentials.json'
client = bigquery.Client(project='mimetic-slice-343817')
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
)

pd.set_option('display.max_colwidth', None)

CURRENT_DIR = os.getcwd()

def create_pyspark_dataframe(df):
    #csv_path = os.path.join(CURRENT_DIR, "data/vi_student_feedbacks.csv")
    #df = pd.read_csv(csv_path)

    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("SparkByExamples.com") \
        .getOrCreate()

    sparkdf=spark.createDataFrame(df)
    return sparkdf

def clean_data(text):
    exclude = set(string.punctuation)
    text =  ''.join(ch for ch in text if ch not in exclude) # remove punctuation
    text = ' '.join(text.split()) # remove extra white space
    text = text.lower() # lower text
    return text

def load_data_from_datalake():
    query_job = client.query(
    """
    SELECT sentence, sentiment FROM `mimetic-slice-343817.vi_student_feedbacks.data` LIMIT 15""")
    results = query_job.result()
    return results.to_dataframe()

def load_cleaned_data():
    query_job = client.query(
    """
    SELECT sentence, sentiment FROM `mimetic-slice-343817.vi_student_feedbacks.data_clean`""")
    results = query_job.result()
    return results.to_dataframe()

def load_unlabeled_data():
    query_job = client.query(
    """
    SELECT sentence, sentiment FROM `mimetic-slice-343817.vi_student_feedbacks.data_unlabeled` LIMIT 10""")
    results = query_job.result()
    return results.to_dataframe()

def push_cleaned_data():
    file_path = 'data/data_clean.csv'
    table_id = 'mimetic-slice-343817.vi_student_feedbacks.data_clean'
    with open(file_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)
    job.result()  # Waits for the job to complete.


def push_unlabeled_data(text):
    file_path = 'data/data_unlabeled.csv'
    
    pd.DataFrame({'sentence': [text], 'sentiment': ''}).to_csv(file_path, index=False)
    
    table_id = 'mimetic-slice-343817.vi_student_feedbacks.data_unlabeled'
    with open(file_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, job_config=job_config)
    job.result()  # Waits for the job to complete.

def clean_data_from_database(isdatalake = False):
    if isdatalake:
        df = load_data_from_datalake()
    else:
        df = load_unlabeled_data()
    
    file_path = 'data/data_clean.csv'

    sparkdf = create_pyspark_dataframe(df)
    cleaning_func = udf(lambda x : clean_data(x), StringType())
    sparkdf.withColumn('sentence', cleaning_func(col('sentence'))).toPandas().to_csv(file_path, index=False)

    push_cleaned_data()


if __name__ == "__main__":
    # print(load_cleaned_data().head())
    #clean_data_from_database(True)
    #print(load_cleaned_data().head())
    #print(load_unlabeled_data().head())

    # file_path = 'data/data_clean.csv'
    # df = pd.read_csv('data/vi_student_feedbacks.csv')
    # sparkdf = create_pyspark_dataframe(df)
    # cleaning_func = udf(lambda x : clean_data(x), StringType())
    # sparkdf.withColumn('sentence', cleaning_func(col('sentence'))).toPandas().to_csv(file_path, index=False)

    print(load_data_from_datalake().head(15))