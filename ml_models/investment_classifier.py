import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql_interface.database_manager import DatabaseManager

class InvestmentClassifier:
    def __init__(self, db_path="data/financial_data.db"):
        self.db = DatabaseManager(db_path)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def get_data(self):
        """Fetch all data from database."""
        # We need all data to calculate aggregate metrics
        conn = self.db.get_latest_prices().head(1) # Just to check connection? No, use get_stock_data
        # Actually, we need to iterate over all tickers.
        # Let's get unique tickers first.
        # Since we don't have a get_tickers method, we can query distinct tickers.
        import sqlite3
        conn = sqlite3.connect(self.db.db_path)
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM stock_data", conn)['ticker'].tolist()
        conn.close()
        return tickers

    def calculate_features(self, ticker):
        """
        Calculate 17 aggregate features for a single ticker.
        
        Features:
        - Total Return
        - Recent Returns (7-day, 30-day)
        - Average RSI, Current RSI
        - Volatility, Sharpe Ratio
        - Price trends (MA7 vs MA30, MA30 vs MA90)
        """
        df = self.db.get_stock_data(ticker)
        if df.empty:
            return None
            
        # Ensure sorted by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'ma_7', 'ma_30', 'ma_90', 'rsi', 'volatility', 'daily_return', 'sharpe_ratio']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        
        # 1. Total Return (First to Last)
        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        total_return = (last_price - first_price) / first_price
        
        # 2. Recent Returns
        # 7-day
        if len(df) >= 7:
            price_7d_ago = df['close'].iloc[-7]
            return_7d = (last_price - price_7d_ago) / price_7d_ago
        else:
            return_7d = 0
            
        # 30-day
        if len(df) >= 30:
            price_30d_ago = df['close'].iloc[-30]
            return_30d = (last_price - price_30d_ago) / price_30d_ago
        else:
            return_30d = 0
            
        # 3. RSI
        avg_rsi = df['rsi'].mean()
        current_rsi = df['rsi'].iloc[-1]
        
        # 4. Volatility & Sharpe
        volatility = df['volatility'].iloc[-1] # Using the latest rolling volatility
        sharpe = df['sharpe_ratio'].iloc[-1] # Using the latest rolling sharpe
        
        # 5. Trends
        # Compare latest MAs
        ma7 = df['ma_7'].iloc[-1]
        ma30 = df['ma_30'].iloc[-1]
        ma90 = df['ma_90'].iloc[-1]
        
        trend_7_30 = 1 if ma7 > ma30 else 0
        trend_30_90 = 1 if ma30 > ma90 else 0
        
        # Additional features to reach 17?
        # The prompt lists: "Total Return, Recent Returns (7, 30), Avg RSI, Curr RSI, Volatility, Sharpe, Trends (2)"
        # That's 1 + 2 + 2 + 2 + 2 = 9 features.
        # "Total: 17 features".
        # Maybe we add more lags or stats?
        # Let's add:
        # - Max Drawdown
        # - Avg Volume
        # - Price vs MA7, Price vs MA30, Price vs MA90
        # - RSI trend (Curr RSI - Avg RSI)
        # - Volatility trend (Curr Vol - Avg Vol)
        
        max_price = df['close'].max()
        min_price = df['close'].min()
        max_drawdown = (min_price - max_price) / max_price
        
        avg_volume = df['volume'].mean()
        
        price_vs_ma7 = (last_price - ma7) / ma7
        price_vs_ma30 = (last_price - ma30) / ma30
        price_vs_ma90 = (last_price - ma90) / ma90
        
        rsi_trend = current_rsi - avg_rsi
        
        # We are at 15. Let's add 2 more.
        # - Std Dev of Volume
        # - Days with positive return ratio
        
        vol_std = df['volume'].std()
        positive_days = (df['daily_return'] > 0).mean()
        
        features = {
            'ticker': ticker,
            'total_return': total_return,
            'return_7d': return_7d,
            'return_30d': return_30d,
            'avg_rsi': avg_rsi,
            'current_rsi': current_rsi,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'trend_7_30': trend_7_30,
            'trend_30_90': trend_30_90,
            'max_drawdown': max_drawdown,
            'avg_volume': avg_volume,
            'price_vs_ma7': price_vs_ma7,
            'price_vs_ma30': price_vs_ma30,
            'price_vs_ma90': price_vs_ma90,
            'rsi_trend': rsi_trend,
            'vol_std': vol_std,
            'positive_days': positive_days
        }
        
        return features

    def calculate_score_and_label(self, features):
        """
        Calculate Composite Score and assign Label.
        
        Score = (Total_Return * 0.3) + (Trend_Score * 0.2) +
                (RSI_Score * 0.15) + (Volatility_Score * 0.15) +
                (Sharpe_Score * 0.2)
        """
        # Normalize/Score components to be on 0-10 scale approximately
        
        # Total Return: > 100% = 10, 0% = 5, -50% = 0
        # Let's just map it: score = (return + 0.5) * 10? No.
        # Let's use a simple heuristic mapping.
        total_ret_score = min(max((features['total_return'] * 10) + 5, 0), 10)
        
        # Trend Score: Both up = 10, Mixed = 5, Both down = 0
        trend_score = 0
        if features['trend_7_30'] and features['trend_30_90']:
            trend_score = 10
        elif features['trend_7_30'] or features['trend_30_90']:
            trend_score = 5
        else:
            trend_score = 2
            
        # RSI Score: 30-70 is good (stable)? Or High is good?
        # Usually RSI < 30 is oversold (Buy), > 70 is overbought (Sell).
        # But for "Investment Potential", maybe we want momentum?
        # Let's assume "Healthy" RSI (40-60) is 10, extremes are lower?
        # Or maybe High RSI = Strong Momentum = High Score?
        # Let's go with Momentum: Higher RSI = Higher Score (capped)
        rsi_score = features['current_rsi'] / 10.0
        
        # Volatility Score: Low volatility is good?
        # Let's say Volatility < 0.01 = 10, > 0.05 = 0
        # Invert volatility
        vol_score = max(10 - (features['volatility'] * 200), 0)
        
        # Sharpe Score: > 1 = 10, < 0 = 0
        sharpe_score = min(max(features['sharpe_ratio'] * 5 + 5, 0), 10)
        
        # Composite Score
        score = (total_ret_score * 0.3) + \
                (trend_score * 0.2) + \
                (rsi_score * 0.15) + \
                (vol_score * 0.15) + \
                (sharpe_score * 0.2)
                
        # Label
        if score >= 7:
            label = "High"
        elif score >= 4:
            label = "Medium"
        else:
            label = "Low"
            
        return score, label

    def run_classification(self):
        """Run the full classification pipeline."""
        tickers = self.get_data()
        results = []
        
        for ticker in tickers:
            feats = self.calculate_features(ticker)
            if not feats:
                continue
                
            score, label = self.calculate_score_and_label(feats)
            
            result = feats.copy()
            result['score'] = score
            result['label'] = label
            results.append(result)
            
        df_results = pd.DataFrame(results)
        
        print("\nClassification Results:")
        print(df_results[['ticker', 'score', 'label']])
        
        # Train Model (as per requirement, even if small data)
        if len(df_results) > 1:
            X = df_results.drop(['ticker', 'score', 'label'], axis=1)
            y = df_results['label']
            
            # Encode labels
            # High=2, Medium=1, Low=0
            # But RF handles strings in some versions, but better to encode
            # Actually sklearn requires numbers.
            
            self.model.fit(X, y)
            print("\nRandom Forest Model trained on aggregated data.")
            
            # Feature Importance
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature Importances:")
            for f in range(X.shape[1]):
                print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
                
        return df_results

if __name__ == "__main__":
    classifier = InvestmentClassifier()
    classifier.run_classification()
