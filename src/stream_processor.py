from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType,
    IntegerType,
)
from pyspark.sql.functions import from_json, col


HOST = "localhost"
PORT = 9999


# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("FraudRealTimeStream")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# 2. Define transaction schema
schema = StructType([
    StructField("transaction_id", LongType(), False),
    StructField("time", DoubleType(), True),
    StructField("amount", DoubleType(), True),
    StructField("class", IntegerType(), True),
])


# 3. Connect to socket stream
raw_stream = (
    spark.readStream
    .format("socket")
    .option("host", HOST)
    .option("port", PORT)
    .load()
)


# 4. Convert JSON text into columns
transactions = (
    raw_stream
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)


# 5. Display incoming transactions
query = (
    transactions.writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)
    .start()
)


# 6. Keep Spark streaming
query.awaitTermination()