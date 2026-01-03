import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_stock_data(tickers, start_date, end_date, output_dir="data/stock_data"):
    """
    Download historical stock data for given tickers and save as CSV.
    
    Args:
        tickers (list): List of stock tickers (e.g., ['AAPL', 'MSFT'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            # Download data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Ensure columns are present (sometimes yfinance returns MultiIndex)
            # We want: Date, Open, High, Low, Close, Volume
            # Note: yfinance columns are usually Capitalized.
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{ticker}_stock_data.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {ticker} data to {output_file} ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

if __name__ == "__main__":
    # 5 years of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"Downloading stock data from {start_date} to {end_date}")
    download_stock_data(TICKERS, start_date, end_date)
