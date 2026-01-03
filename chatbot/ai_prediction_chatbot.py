import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import ollama
import os
import sys
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql_interface.database_manager import DatabaseManager
from ml_models.spark_gbt_forecaster import SparkGBTForecaster

class FinancialChatbot:
    def __init__(self):
        self.db = DatabaseManager()
        # Initialize Forecaster (this might take a moment to start Spark)
        # We lazy load it or cache it to avoid restarting Spark on every rerun
        if 'forecaster' not in st.session_state:
            st.session_state.forecaster = SparkGBTForecaster()
        self.forecaster = st.session_state.forecaster

    def get_stock_data(self, ticker, days=30):
        """Fetch historical data."""
        # Calculate start date based on days
        from datetime import datetime, timedelta
        # We don't have a simple "last N days" query in DB manager, so we fetch all and tail
        # Or we can fetch by date range if we knew the dates.
        # Let's just fetch all and take the tail.
        df = self.db.get_stock_data(ticker)
        if df.empty:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df.tail(days)

    def predict_stock(self, ticker, days=7):
        """Predict future prices."""
        try:
            predictions = self.forecaster.predict_future(ticker, num_days=days)
            return pd.DataFrame(predictions)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

    def generate_plot(self, history_df, prediction_df=None, ticker="Stock"):
        """Generate a plot of historical and predicted data."""
        plt.figure(figsize=(10, 5))
        
        # Plot history
        plt.plot(history_df['date'], history_df['close'], label='Historical Close', color='blue')
        
        # Plot prediction
        if prediction_df is not None:
            # Connect last history point to first prediction point for continuity
            last_hist_date = history_df['date'].iloc[-1]
            last_hist_price = history_df['close'].iloc[-1]
            
            pred_dates = [pd.to_datetime(d) for d in prediction_df['Date']]
            pred_prices = prediction_df['Predicted_Close']
            
            # Prepend last history point
            all_pred_dates = [last_hist_date] + pred_dates
            all_pred_prices = [last_hist_price] + list(pred_prices)
            
            plt.plot(all_pred_dates, all_pred_prices, label='Forecast', color='green', linestyle='--')
            
        plt.title(f"{ticker} Price Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def detect_intent(self, query):
        """Detect intent from user query."""
        query = query.lower()
        
        # Extract ticker
        # Simple regex for 1-5 uppercase letters (assuming user types Ticker or we extract from context)
        # But user might type "aapl" (lowercase).
        # Let's look for common tickers or just words that match ticker format if uppercase?
        # Better: Look for known tickers in the database.
        # For now, let's just regex for common ones or assume user specifies it clearly.
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        found_ticker = None
        for t in tickers:
            if t.lower() in query:
                found_ticker = t
                break
        
        # Extract days
        days = 7
        days_match = re.search(r'(\d+)\s*days?', query)
        if days_match:
            days = int(days_match.group(1))
            
        if "predict" in query or "forecast" in query:
            return "prediction", found_ticker, days
        elif "data" in query or "show" in query or "price" in query:
            return "data", found_ticker, days
        else:
            return "general", found_ticker, days

    def generate_llm_response(self, query):
        """Generate response using Ollama."""
        try:
            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'system', 'content': 'You are a helpful financial assistant. Answer questions about finance, stocks, and trading concepts concisely.'},
                {'role': 'user', 'content': query}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error connecting to LLM: {e}. Make sure Ollama is running."

def main():
    st.set_page_config(page_title="AI Financial Chatbot", page_icon="🤖")
    
    st.title("🤖 AI Financial Assistant")
    st.markdown("Ask about stock prices, predictions, or financial concepts!")
    
    # Initialize Chatbot
    if 'bot' not in st.session_state:
        st.session_state.bot = FinancialChatbot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            intent, ticker, days = st.session_state.bot.detect_intent(prompt)
            
            response_text = ""
            response_image = None
            
            if intent == "prediction":
                if ticker:
                    message_placeholder.markdown(f"Generating prediction for **{ticker}** for the next {days} days...")
                    preds = st.session_state.bot.predict_stock(ticker, days)
                    hist = st.session_state.bot.get_stock_data(ticker, 30)
                    
                    if preds is not None and hist is not None:
                        response_text = f"Here is the price forecast for **{ticker}**."
                        response_image = st.session_state.bot.generate_plot(hist, preds, ticker)
                        st.image(response_image)
                        st.dataframe(preds)
                    else:
                        response_text = f"Could not generate prediction for {ticker}."
                else:
                    response_text = "Please specify a ticker (e.g., AAPL, MSFT) for prediction."
                    
            elif intent == "data":
                if ticker:
                    hist = st.session_state.bot.get_stock_data(ticker, days)
                    if hist is not None:
                        response_text = f"Here is the recent data for **{ticker}**."
                        response_image = st.session_state.bot.generate_plot(hist, ticker=ticker)
                        st.image(response_image)
                        st.dataframe(hist.tail())
                    else:
                        response_text = f"No data found for {ticker}."
                else:
                    response_text = "Please specify a ticker to view data."
                    
            else:
                # General conversation
                with st.spinner("Thinking..."):
                    response_text = st.session_state.bot.generate_llm_response(prompt)
            
            message_placeholder.markdown(response_text)
            
            # Save to history
            msg_data = {"role": "assistant", "content": response_text}
            if response_image:
                msg_data["image"] = response_image
            st.session_state.messages.append(msg_data)

if __name__ == "__main__":
    main()
