import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, lit, rand, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import numpy as np

class SparkGBTForecaster:
    def __init__(self, data_path="data/processed_stocks.parquet", model_path="ml_models/gbt_model"):
        self.data_path = data_path
        self.model_path = model_path
        self.spark = SparkSession.builder \
            .appName("StockForecasting") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def create_lagged_features(self, df, lags=30):
        """Create lagged features for time series forecasting."""
        w = Window.partitionBy("Ticker").orderBy("Date")
        
        feature_cols = []
        
        # Create lags for OHLCV
        cols_to_lag = ["Close", "Open", "High", "Low", "Volume"]
        
        # We need to be careful with stack depth or plan size, so maybe do it iteratively or in batches?
        # The requirement says "Create features in batches (1-10, 11-20, 21-30) to avoid StackOverflowError"
        
        for i in range(1, lags + 1):
            for c in cols_to_lag:
                col_name = f"{c}_lag_{i}"
                df = df.withColumn(col_name, lag(c, i).over(w))
                feature_cols.append(col_name)
            
            # Cache every 10 lags to break lineage
            if i % 10 == 0:
                df = df.cache()
                df.count() # Force materialization
        
        # Create Target: Close price 7 days in future
        df = df.withColumn("Target", lag("Close", -7).over(w))
        
        # Drop rows with nulls (due to lags and future target)
        df = df.na.drop()
        
        return df, feature_cols

    def train_model(self):
        """Train GBT Regressor."""
        print("Loading data...")
        df = self.spark.read.parquet(self.data_path)
        
        print("Creating features (this may take a while)...")
        df_processed, feature_cols = self.create_lagged_features(df)
        
        print(f"Data shape after feature engineering: {df_processed.count()} rows")
        
        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # Split data
        # We should split by time, but random split is requested in requirements: "Split: 80% train, 10% validation, 10% test"
        # Time series split is better, but let's follow requirements or standard practice.
        # Standard practice for time series is NOT random split.
        # However, the requirement explicitly says "Split: 80% train, 10% validation, 10% test".
        # I will use random split as per requirement, but note it's not ideal for time series.
        # Actually, let's try to respect time if possible, but random split is easier with Spark.
        
        train_data, val_data, test_data = df_processed.randomSplit([0.8, 0.1, 0.1], seed=42)
        
        # Define GBT
        gbt = GBTRegressor(featuresCol="features", labelCol="Target", maxIter=100, maxDepth=6, stepSize=0.1, subsamplingRate=0.8)
        
        # Pipeline
        pipeline = Pipeline(stages=[assembler, gbt])
        
        print("Training model...")
        model = pipeline.fit(train_data)
        
        # Evaluate
        print("Evaluating model...")
        predictions = model.transform(test_data)
        
        evaluator_rmse = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions)
        
        evaluator_r2 = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="r2")
        r2 = evaluator_r2.evaluate(predictions)
        
        evaluator_mae = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="mae")
        mae = evaluator_mae.evaluate(predictions)
        
        print(f"Model Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R2: {r2:.2f}")
        
        # Save model
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return model

    def predict_future(self, ticker, num_days=7):
        """Predict future stock prices."""
        # Load model if not passed
        from pyspark.ml import PipelineModel
        model = PipelineModel.load(self.model_path) # PipelineModel
        
        # Get latest data for ticker
        df = self.spark.read.parquet(self.data_path).filter(col("Ticker") == ticker)
        
        # We need the last 30 days to predict the next day.
        # But to predict N days recursively, we need to append predictions and shift window.
        # This is complex in Spark.
        # Simplified approach:
        # Just predict the next 7 days based on the LATEST available window?
        # No, the model predicts "Target" which is T+7.
        # So if we feed it T (today), it predicts T+7.
        # But the requirement says "Predict next N days".
        # And "predict_future(ticker, num_days) to forecast next N days".
        # And "Add realistic variation (1% std deviation)".
        
        # If the model predicts T+7 directly, then for T+1...T+6 we don't have a direct model?
        # Or maybe the model is trained to predict T+1?
        # Requirement: "Target: Predict Close price 7 days into the future".
        # This implies it predicts ONE value: Price at T+7.
        # If so, how do we get T+1, T+2?
        # Maybe we should have trained it for T+1?
        # Or maybe the requirement means "Predict the price sequence for next 7 days"?
        # "Predict next N days" usually means a sequence.
        # If the model only outputs T+7, we can only predict T+7.
        # Unless we treat it as a generic "future predictor" and apply it iteratively?
        # But it's trained on 30 lags to predict T+7.
        
        # Let's assume the requirement meant "Predict T+1" and iterate?
        # "Target: Predict Close price 7 days into the future" is specific.
        # Maybe it means we want to know the price in a week.
        # But "predict_future(ticker, num_days)" suggests a sequence.
        
        # Let's stick to the requirement: "Target: Predict Close price 7 days into the future".
        # And for "predict_future(ticker, num_days)", maybe we just return the prediction for T+7?
        # Or maybe we use the last 7 days of data to predict T+7, T+8... T+13?
        # i.e. Input(T) -> Output(T+7).
        # Input(T-1) -> Output(T+6).
        # Input(T-6) -> Output(T+1).
        # So we can generate predictions for T+1 to T+7 by using inputs from T-6 to T.
        # Yes! That makes sense.
        # We use the sliding window ending at T, T-1, ..., T-6 to predict T+7, T+6, ..., T+1.
        
        # Get last num_days rows
        w = Window.partitionBy("Ticker").orderBy("Date")
        df_with_row = df.withColumn("row_num", row_number().over(w.orderBy(col("Date").desc())))
        
        # We need the last num_days rows to generate predictions for next num_days?
        # Wait.
        # Model: F(History_30) -> Price(T+7).
        # We want Price(T+1).
        # We need F(History_30 ending at T-6).
        # Because if History ends at T-6, then "7 days into future" is T+1.
        # Correct.
        
        # So to predict T+1...T+7, we need inputs ending at T-6...T.
        # So we need the last 7 rows of the dataset, and for each, we construct the 30-day history ENDING at that row.
        # Then we predict.
        
        # Let's get the last num_days rows.
        last_rows = df_with_row.filter(col("row_num") <= num_days).orderBy("Date")
        
        # We need to construct features for these rows.
        # The `create_lagged_features` function does this for the whole DF.
        # So we can just reuse it on the whole DF (or filtered DF), then filter for the last num_days rows.
        # But `create_lagged_features` shifts data.
        # If we have row at T. `create_lagged_features` creates lags T-1...T-30.
        # And Target is T+7.
        # So if we apply model to row T, we get prediction for T+7.
        # If we apply to row T-1, we get prediction for T+6.
        # ...
        # If we apply to row T-6, we get prediction for T+1.
        
        # So we need to take the last num_days rows from the feature-engineered DF.
        # And their predictions will be for T+1 to T+num_days relative to the *start of the sequence*?
        # No.
        # Row T: Prediction is T+7.
        # Row T-1: Prediction is T+6.
        # ...
        # Row T-(num_days-1): Prediction is T+7 - (num_days-1).
        
        # If num_days=7.
        # Row T: Pred T+7.
        # Row T-6: Pred T+1.
        
        # So yes, we just need the last num_days rows of the feature-engineered dataframe.
        
        df_processed, _ = self.create_lagged_features(df)
        
        # Assemble
        assembler = VectorAssembler(inputCols=model.stages[0].getInputCols(), outputCol="features")
        df_assembled = assembler.transform(df_processed)
        
        # Predict
        preds = model.stages[1].transform(df_assembled)
        
        # Get last num_days predictions
        # We need to sort by Date
        preds_collected = preds.select("Date", "prediction").orderBy(col("Date").desc()).limit(num_days).collect()
        
        # The predictions are in reverse order (T -> T+7, T-1 -> T+6, ...).
        # And we want to return them as a sequence for next 1..7 days.
        # preds_collected[0] is from Row T, so it's prediction for T+7.
        # preds_collected[6] is from Row T-6, so it's prediction for T+1.
        
        results = []
        import datetime
        
        # Get the actual last date in data
        last_date_row = df.agg({"Date": "max"}).collect()[0]
        last_date = last_date_row[0] if last_date_row else datetime.datetime.now()
        if isinstance(last_date, str):
             last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
        
        # Sort predictions by date ascending (T-6 to T)
        preds_collected.sort(key=lambda x: x['Date'])
        
        # Now preds_collected[0] is T-6 -> Pred T+1
        # preds_collected[6] is T -> Pred T+7
        
        for i, row in enumerate(preds_collected):
            predicted_price = row['prediction']
            # Add random variation as requested
            variation = np.random.normal(0, predicted_price * 0.01)
            final_price = predicted_price + variation
            
            future_date = last_date + datetime.timedelta(days=i+1)
            results.append({
                "Date": future_date.strftime("%Y-%m-%d"),
                "Predicted_Close": final_price
            })
            
        return results

if __name__ == "__main__":
    forecaster = SparkGBTForecaster()
    forecaster.train_model()
    
    # Test prediction
    print("\nPredictions for AAPL:")
    preds = forecaster.predict_future("AAPL", 7)
    for p in preds:
        print(p)
