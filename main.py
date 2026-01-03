"""
Main Pipeline Orchestrator - AI Financial Analysis Platform

STUDENT TASK:
Complete this main pipeline file to orchestrate all components

USAGE:
    python main.py

Then select from menu:
1. Data Collection
2. Preprocessing
3. Database Setup
4. Train ML Models
5. Run Chatbot
6. Run Dashboard
7. Run Complete Pipeline
"""

import os
import sys
from pyspark.sql import SparkSession

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *


def main():
    """
    Main menu loop
    """
    while True:
        print("\n" + "="*50)
        print("AI Financial Analysis Platform")
        print("="*50)
        print("1. Data Collection (Download Stock Data)")
        print("2. Preprocessing (Spark Feature Engineering)")
        print("3. Database Setup (Load Data to SQLite)")
        print("4. Train ML Models (GBT Forecasting)")
        print("5. Run Investment Classification")
        print("6. Run AI Chatbot")
        print("7. Run Dashboard")
        print("8. Run Tests")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ")
        
        if choice == '1':
            print("\n--- Data Collection ---")
            from data_collection.stock_downloader import download_stock_data, TICKERS
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            download_stock_data(TICKERS, start_date, end_date)
            
        elif choice == '2':
            print("\n--- Preprocessing ---")
            from preprocessing.spark_preprocessor import preprocess_data
            preprocess_data()
            
        elif choice == '3':
            print("\n--- Database Setup ---")
            from sql_interface.database_manager import DatabaseManager
            db = DatabaseManager()
            db.load_data_from_parquet()
            
        elif choice == '4':
            print("\n--- Train ML Models ---")
            from ml_models.spark_gbt_forecaster import SparkGBTForecaster
            forecaster = SparkGBTForecaster()
            forecaster.train_model()
            
        elif choice == '5':
            print("\n--- Investment Classification ---")
            from ml_models.investment_classifier import InvestmentClassifier
            classifier = InvestmentClassifier()
            classifier.run_classification()
            
        elif choice == '6':
            print("\n--- AI Chatbot ---")
            print("Starting Streamlit Chatbot...")
            os.system("streamlit run chatbot/ai_prediction_chatbot.py")
            
        elif choice == '7':
            print("\n--- Dashboard ---")
            print("Starting Streamlit Dashboard...")
            os.system("streamlit run dashboard/dashboard_app.py")
            
        elif choice == '8':
            print("\n--- Running Tests ---")
            os.system("pytest tests/test_pipeline.py -v")
            
        elif choice == '0':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
