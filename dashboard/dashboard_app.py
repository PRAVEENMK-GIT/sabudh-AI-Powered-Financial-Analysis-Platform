import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql_interface.database_manager import DatabaseManager
from ml_models.spark_gbt_forecaster import SparkGBTForecaster
from ml_models.investment_classifier import InvestmentClassifier

# Set page config
st.set_page_config(page_title="Financial Analysis Platform", layout="wide")

class DashboardApp:
    def __init__(self):
        self.db = DatabaseManager()
        if 'forecaster' not in st.session_state:
            st.session_state.forecaster = SparkGBTForecaster()
        self.forecaster = st.session_state.forecaster
        
        if 'classifier' not in st.session_state:
            st.session_state.classifier = InvestmentClassifier()
        self.classifier = st.session_state.classifier

    def load_tickers(self):
        # Get tickers from DB
        import sqlite3
        conn = sqlite3.connect(self.db.db_path)
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM stock_data", conn)['ticker'].tolist()
        conn.close()
        return tickers

    def render_stock_viewer(self, tickers):
        st.header("Stock Data Viewer")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_ticker = st.selectbox("Select Ticker", tickers, key="viewer_ticker")
            days = st.slider("Number of days to view", 30, 365*5, 365)
            
        df = self.db.get_stock_data(selected_ticker)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(days)
            
            with col2:
                st.subheader(f"{selected_ticker} Closing Price")
                st.line_chart(df.set_index('date')['close'])
            
            st.dataframe(df)
        else:
            st.warning("No data found.")

    def render_indicators(self, tickers):
        st.header("Technical Indicators")
        selected_ticker = st.selectbox("Select Ticker", tickers, key="ind_ticker")
        
        df = self.db.get_stock_data(selected_ticker)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(180) # Last 6 months
            
            # Moving Averages
            st.subheader("Moving Averages")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['date'], df['close'], label='Close', alpha=0.5)
            ax.plot(df['date'], df['ma_7'], label='MA 7')
            ax.plot(df['date'], df['ma_30'], label='MA 30')
            ax.plot(df['date'], df['ma_90'], label='MA 90')
            ax.legend()
            st.pyplot(fig)
            
            # RSI
            st.subheader("Relative Strength Index (RSI)")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(df['date'], df['rsi'], color='purple')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_ylim(0, 100)
            st.pyplot(fig2)
            
            # Volatility
            st.subheader("Volatility (30-day Rolling)")
            st.line_chart(df.set_index('date')['volatility'])

    def render_predictions(self, tickers):
        st.header("ML Price Prediction (GBT)")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_ticker = st.selectbox("Select Ticker", tickers, key="pred_ticker")
            pred_days = st.slider("Days to Predict", 1, 30, 7)
            if st.button("Generate Prediction"):
                with st.spinner("Running Spark Model..."):
                    preds = self.forecaster.predict_future(selected_ticker, pred_days)
                    st.session_state.preds = preds
                    st.session_state.last_pred_ticker = selected_ticker
        
        if 'preds' in st.session_state and st.session_state.get('last_pred_ticker') == selected_ticker:
            preds = st.session_state.preds
            pred_df = pd.DataFrame(preds)
            
            with col2:
                st.subheader(f"Forecast for {selected_ticker}")
                
                # Get history for context
                hist = self.db.get_stock_data(selected_ticker).tail(60)
                hist['date'] = pd.to_datetime(hist['date'])
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(hist['date'], hist['close'], label='History')
                
                # Plot prediction
                pred_dates = [pd.to_datetime(d) for d in pred_df['Date']]
                ax.plot(pred_dates, pred_df['Predicted_Close'], label='Forecast', linestyle='--', marker='o')
                
                ax.legend()
                st.pyplot(fig)
            
            st.write("Predicted Values:")
            st.dataframe(pred_df)

    def render_classification(self):
        st.header("Investment Classification")
        
        if st.button("Run Classification Analysis"):
            with st.spinner("Analyzing all stocks..."):
                results = self.classifier.run_classification()
                st.session_state.class_results = results
        
        if 'class_results' in st.session_state:
            results = st.session_state.class_results
            
            # Highlight High potential
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            high = results[results['label'] == 'High']
            medium = results[results['label'] == 'Medium']
            low = results[results['label'] == 'Low']
            
            col1.metric("High Potential", len(high))
            col2.metric("Medium Potential", len(medium))
            col3.metric("Low Potential", len(low))
            
            st.subheader("Detailed Analysis")
            
            # Color code
            def color_label(val):
                color = 'green' if val == 'High' else 'orange' if val == 'Medium' else 'red'
                return f'color: {color}'
            
            st.dataframe(results.style.applymap(color_label, subset=['label']))

    def render_explanations(self):
        st.header("Model Explanations")
        
        st.subheader("1. Time Series Forecasting (Spark GBT)")
        st.markdown("""
        **Algorithm**: Gradient Boosted Trees Regressor
        - **Input**: 150 features (30 days of lagged history for Open, High, Low, Close, Volume)
        - **Target**: Close price 7 days in the future
        - **Why GBT?**: Handles non-linear relationships and feature interactions well.
        """)
        
        st.subheader("2. Investment Classification (Random Forest)")
        st.markdown("""
        **Algorithm**: Random Forest Classifier
        - **Input**: 17 aggregate features (Returns, RSI, Volatility, Trends)
        - **Logic**: Composite Score based on weighted factors.
        - **Labels**: High (Score >= 7), Medium (4-7), Low (< 4)
        """)
        
        st.info("This platform uses PySpark for data processing and training, ensuring scalability for large datasets.")

def main():
    st.title("📈 AI Financial Analysis Platform")
    
    app = DashboardApp()
    tickers = app.load_tickers()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stock Data", 
        "Technical Indicators", 
        "ML Predictions", 
        "Classification",
        "Explanations"
    ])
    
    with tab1:
        app.render_stock_viewer(tickers)
        
    with tab2:
        app.render_indicators(tickers)
        
    with tab3:
        app.render_predictions(tickers)
        
    with tab4:
        app.render_classification()
        
    with tab5:
        app.render_explanations()

if __name__ == "__main__":
    main()
