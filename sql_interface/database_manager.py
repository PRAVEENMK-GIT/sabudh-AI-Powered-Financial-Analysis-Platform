import sqlite3
import pandas as pd
import os

class DatabaseManager:
    def __init__(self, db_path="data/financial_data.db"):
        self.db_path = db_path
        self.create_tables()

    def create_tables(self):
        """Create the stock_data table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT NOT NULL,
            date DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            ma_7 REAL,
            ma_30 REAL,
            ma_90 REAL,
            rsi REAL,
            volatility REAL,
            daily_return REAL,
            sharpe_ratio REAL,
            PRIMARY KEY (ticker, date)
        );
        """
        cursor.execute(create_table_sql)
        
        # Create index for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (ticker, date);")
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")

    def load_data_from_parquet(self, parquet_path="data/processed_stocks.parquet"):
        """Load processed data from Parquet file into SQLite."""
        if not os.path.exists(parquet_path):
            print(f"Parquet file not found: {parquet_path}")
            return
            
        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Ensure column names match database schema (lowercase)
        df.columns = [c.lower() for c in df.columns]
        
        conn = sqlite3.connect(self.db_path)
        try:
            # Write to database
            df.to_sql("stock_data", conn, if_exists="replace", index=False)
            
            # Re-create index since 'replace' drops the table
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (ticker, date);")
            
            print(f"Successfully loaded {len(df)} rows into database.")
        except Exception as e:
            print(f"Error loading data: {e}")
        finally:
            conn.close()

    def get_stock_data(self, ticker, start_date=None, end_date=None):
        """Retrieve stock data for a specific ticker."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM stock_data WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_latest_prices(self):
        """Get the latest available data for all tickers."""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT t1.*
        FROM stock_data t1
        INNER JOIN (
            SELECT ticker, MAX(date) as max_date
            FROM stock_data
            GROUP BY ticker
        ) t2 ON t1.ticker = t2.ticker AND t1.date = t2.max_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

if __name__ == "__main__":
    db = DatabaseManager()
    db.load_data_from_parquet()
    
    # Test queries
    print("\nLatest Prices:")
    print(db.get_latest_prices()[['ticker', 'date', 'close']])
    
    print("\nSample AAPL Data:")
    print(db.get_stock_data("AAPL").head())
