from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pandas as pd
import string
import os

CURRENT_DIR = os.getcwd()

def create_pyspark_dataframe():
    csv_path = os.path.join(CURRENT_DIR, "vi_student_feedbacks.csv")
    df = pd.read_csv(csv_path)

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

if __name__ == "__main__":
    sparkdf = create_pyspark_dataframe()
    cleaning_func = udf(lambda x : clean_data(x), StringType())
    sparkdf.withColumn('sentence', cleaning_func(col('sentence'))).show(10, truncate=False)
