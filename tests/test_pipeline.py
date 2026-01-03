import pytest
import pandas as pd
import numpy as np
import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_interface.database_manager import DatabaseManager
from ml_models.investment_classifier import InvestmentClassifier

# We skip Spark tests in this simple suite to avoid overhead/complexity in CI environments without Spark setup,
# but we can add a simple one if needed.
# For now, let's test the components that are easier to test in isolation.

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_path = "tests/test_financial_data.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = DatabaseManager(db_path)
    yield db
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

def test_database_creation(temp_db):
    """Test if database tables are created correctly."""
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    cursor = conn.cursor()
    
    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data';")
    assert cursor.fetchone() is not None
    
    # Check columns
    cursor.execute("PRAGMA table_info(stock_data);")
    columns = [info[1] for info in cursor.fetchall()]
    expected_cols = ['ticker', 'date', 'open', 'close', 'ma_7', 'rsi']
    for col in expected_cols:
        assert col in columns
        
    conn.close()

def test_database_insertion(temp_db):
    """Test inserting and retrieving data."""
    # Create dummy data
    data = {
        'ticker': ['TEST'],
        'date': [pd.Timestamp('2023-01-01')],
        'open': [100.0],
        'high': [105.0],
        'low': [95.0],
        'close': [102.0],
        'volume': [1000],
        'ma_7': [100.0],
        'ma_30': [98.0],
        'ma_90': [95.0],
        'rsi': [50.0],
        'volatility': [0.02],
        'daily_return': [0.01],
        'sharpe_ratio': [1.5]
    }
    df = pd.DataFrame(data)
    
    # Insert manually (since load_data_from_parquet expects a file)
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    df.to_sql("stock_data", conn, if_exists="append", index=False)
    conn.close()
    
    # Retrieve
    retrieved_df = temp_db.get_stock_data("TEST")
    assert len(retrieved_df) == 1
    assert retrieved_df.iloc[0]['close'] == 102.0

def test_investment_classifier_logic():
    """Test the scoring logic of the classifier."""
    # Mock features
    features = {
        'ticker': 'TEST',
        'total_return': 0.5,      # 50% return -> Score ~ 10
        'return_7d': 0.01,
        'return_30d': 0.05,
        'avg_rsi': 50,
        'current_rsi': 60,        # 60/10 = 6
        'volatility': 0.01,       # Low vol -> High score
        'sharpe_ratio': 1.0,      # Good sharpe -> High score
        'trend_7_30': 1,
        'trend_30_90': 1,         # Strong trend -> 10
        'max_drawdown': -0.1,
        'avg_volume': 100000,
        'price_vs_ma7': 0.01,
        'price_vs_ma30': 0.02,
        'price_vs_ma90': 0.05,
        'rsi_trend': 10,
        'vol_std': 0.001,
        'positive_days': 0.55
    }
    
    # We can use the class method directly if we instantiate it
    # But the class requires a DB path in init.
    # Let's mock the DB or just use a dummy path.
    clf = InvestmentClassifier(db_path="dummy.db")
    
    score, label = clf.calculate_score_and_label(features)
    
    # Check logic
    # Total Return Score: min(max((0.5 * 10) + 5, 0), 10) = 10
    # Trend Score: 10
    # RSI Score: 6
    # Vol Score: max(10 - (0.01 * 200), 0) = 8
    # Sharpe Score: min(max(1.0 * 5 + 5, 0), 10) = 10
    
    # Weighted:
    # 10*0.3 + 10*0.2 + 6*0.15 + 8*0.15 + 10*0.2
    # 3.0 + 2.0 + 0.9 + 1.2 + 2.0 = 9.1
    
    assert score > 8.0
    assert label == "High"

def test_investment_classifier_low_score():
    """Test logic for a low performing stock."""
    features = {
        'ticker': 'BAD',
        'total_return': -0.5,     # -50% -> Score 0
        'return_7d': -0.01,
        'return_30d': -0.05,
        'avg_rsi': 40,
        'current_rsi': 30,        # 3
        'volatility': 0.05,       # High vol -> 10 - 10 = 0
        'sharpe_ratio': -1.0,     # Bad sharpe -> 0
        'trend_7_30': 0,
        'trend_30_90': 0,         # No trend -> 2
        'max_drawdown': -0.5,
        'avg_volume': 100000,
        'price_vs_ma7': -0.01,
        'price_vs_ma30': -0.02,
        'price_vs_ma90': -0.05,
        'rsi_trend': -10,
        'vol_std': 0.005,
        'positive_days': 0.4
    }
    
    clf = InvestmentClassifier(db_path="dummy.db")
    score, label = clf.calculate_score_and_label(features)
    
    # Weighted:
    # 0*0.3 + 2*0.2 + 3*0.15 + 0*0.15 + 0*0.2
    # 0 + 0.4 + 0.45 + 0 + 0 = 0.85
    
    assert score < 4.0
    assert label == "Low"
