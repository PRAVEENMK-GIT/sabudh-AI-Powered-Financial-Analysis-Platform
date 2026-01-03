import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, avg, stddev, lag, when, row_number
from pyspark.sql.window import Window
import pyspark.sql.functions as F

def preprocess_data(input_dir="data/stock_data", output_file="data/processed_stocks.parquet"):
    """
    Load stock data, perform feature engineering, and save as Parquet.
    """
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("StockDataPreprocessing") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("Spark session initialized.")
    
    # Load and merge data
    dataframes = []
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    for file in files:
        ticker = file.split("_")[0]
        file_path = os.path.join(input_dir, file)
        
        # Read CSV
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Add Ticker column
        df = df.withColumn("Ticker", lit(ticker))
        
        # Cast columns to Double
        for col_name in ["Open", "High", "Low", "Close", "Volume"]:
            df = df.withColumn(col_name, col(col_name).cast("double"))
        
        dataframes.append(df)
    
    if not dataframes:
        print("No data found.")
        return

    # Union all dataframes
    full_df = dataframes[0]
    for df in dataframes[1:]:
        full_df = full_df.union(df)
        
    print(f"Loaded {full_df.count()} rows.")
    
    # Define Window specifications
    # Order by Date for each Ticker
    w = Window.partitionBy("Ticker").orderBy("Date")
    w_rolling_7 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-6, 0)
    w_rolling_30 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-29, 0)
    w_rolling_90 = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-89, 0)
    
    # 1. Moving Averages
    full_df = full_df.withColumn("MA_7", avg("Close").over(w_rolling_7)) \
                     .withColumn("MA_30", avg("Close").over(w_rolling_30)) \
                     .withColumn("MA_90", avg("Close").over(w_rolling_90))
    
    # 2. Daily Returns
    # Return = (Close - Prev_Close) / Prev_Close
    full_df = full_df.withColumn("Prev_Close", lag("Close", 1).over(w))
    full_df = full_df.withColumn("Daily_Return", (col("Close") - col("Prev_Close")) / col("Prev_Close"))
    
    # 3. Volatility (30-day rolling std dev of Daily Returns)
    full_df = full_df.withColumn("Volatility", stddev("Daily_Return").over(w_rolling_30))
    
    # 4. Sharpe Ratio (Rolling 30-day Mean Return / Rolling 30-day Std Dev)
    # Assuming risk-free rate is 0 for simplicity or just Mean/StdDev as requested
    full_df = full_df.withColumn("Mean_Return_30", avg("Daily_Return").over(w_rolling_30))
    full_df = full_df.withColumn("Sharpe_Ratio", col("Mean_Return_30") / col("Volatility"))
    
    # 5. RSI (14-day)
    # RSI = 100 - (100 / (1 + RS))
    # RS = Avg Gain / Avg Loss
    
    # Calculate Change
    full_df = full_df.withColumn("Change", col("Close") - col("Prev_Close"))
    
    # Separate Gain and Loss
    full_df = full_df.withColumn("Gain", when(col("Change") > 0, col("Change")).otherwise(0))
    full_df = full_df.withColumn("Loss", when(col("Change") < 0, -col("Change")).otherwise(0))
    
    # Calculate Avg Gain and Avg Loss (Simple Moving Average for RSI usually, or Wilder's)
    # Using Simple Moving Average for simplicity as per common implementations unless Wilder's is specified.
    # Standard RSI often uses Wilder's Smoothing, but SMA is also common in simple implementations.
    # Let's use SMA over 14 days.
    w_rsi = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-13, 0)
    
    full_df = full_df.withColumn("Avg_Gain", avg("Gain").over(w_rsi))
    full_df = full_df.withColumn("Avg_Loss", avg("Loss").over(w_rsi))
    
    # Calculate RS
    # Avoid division by zero
    full_df = full_df.withColumn("RS", when(col("Avg_Loss") == 0, 100).otherwise(col("Avg_Gain") / col("Avg_Loss")))
    
    # Calculate RSI
    full_df = full_df.withColumn("RSI", 100 - (100 / (1 + col("RS"))))
    
    # Handle missing values (first few rows will have nulls due to lags/windows)
    # Requirement: "Handle missing values (forward fill or drop)"
    # Since we need valid data for ML, dropping initial rows is safer than forward filling zeros for things like MA.
    # However, let's drop rows with nulls in critical columns.
    full_df = full_df.na.drop()
    
    # Select final columns
    final_columns = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", 
                     "MA_7", "MA_30", "MA_90", "RSI", "Volatility", "Daily_Return", "Sharpe_Ratio"]
    
    final_df = full_df.select(final_columns)
    
    print(f"Processed {final_df.count()} rows.")
    
    # Save as Parquet
    final_df.write.mode("overwrite").parquet(output_file)
    print(f"Saved processed data to {output_file}")
    
    spark.stop()

if __name__ == "__main__":
    preprocess_data()
