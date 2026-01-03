from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SchemaCheck") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/processed_stocks.parquet")
df.printSchema()
df.show(5)
