# AI-Powered Financial Analysis Platform - Student Assignment

## 📋 Project Overview

This is a comprehensive Data Engineering and Machine Learning project that builds an **AI-Powered Financial Analysis Platform** using PySpark, Machine Learning, and LLMs.

### What You'll Build:
1. **Data Collection Pipeline** - Fetch stock data from Yahoo Finance
2. **Data Preprocessing** - Clean and engineer features using PySpark
3. **SQLite Database** - Store processed data
4. **ML Time Series Forecasting** - Predict stock prices using Spark MLlib GBT (Gradient Boosted Trees)
5. **ML Classification** - Classify investment potential (High/Medium/Low)
6. **AI Chatbot** - Interactive chatbot with database queries, ML predictions, and graph generation
7. **Dashboard** - Streamlit dashboard for visualization

---

## 🎯 Learning Objectives

By completing this project, you will learn:

- ✅ **PySpark** - Distributed data processing and feature engineering
- ✅ **Machine Learning** - Time series forecasting and classification
- ✅ **Spark MLlib** - Gradient Boosted Trees (GBT) for regression
- ✅ **SQL** - Database design and queries
- ✅ **LLMs** - Integration with Llama models via Ollama
- ✅ **Data Engineering** - ETL pipelines and data validation
- ✅ **Web Development** - Streamlit dashboards and chatbots
- ✅ **Docker** - Containerization for LLM deployment

---

## 📦 Tech Stack

### Core Technologies:
- **Python 3.8+** - Primary programming language
- **PySpark 3.5+** - Distributed data processing
- **Spark MLlib** - Machine learning (GBT Regressor for forecasting)
- **SQLite3** - Relational database
- **Pandas** - Data manipulation
- **yfinance** - Stock data API
- **Streamlit** - Web dashboards and chatbot UI
- **Matplotlib** - Data visualization and graph generation

### LLM Stack:
- **Ollama** - Local LLM runtime (runs in Docker)
- **Llama 3.2** - Language model for chatbot conversations
- **Docker** - Container for Ollama

### ML Models:
1. **Spark GBT Forecaster** (Gradient Boosted Trees) - Time series prediction
2. **Random Forest Classifier** - Investment classification

---

## 🏗️ Project Structure

```
project_template/
├── README.md                          # This file
├── SETUP.md                           # Detailed setup instructions
├── requirements.txt                   # Python dependencies
├── main.py                            # Main pipeline orchestrator
│
├── config/
│   ├── __init__.py
│   └── config.py                      # Configuration settings
│
├── data_collection/
│   ├── __init__.py
│   ├── stock_downloader.py           # Yahoo Finance data collection
│   └── sec_downloader.py             # SEC 10-K filings downloader (optional)
│
├── preprocessing/
│   ├── __init__.py
│   └── spark_preprocessor.py         # PySpark feature engineering
│
├── sql_interface/
│   ├── __init__.py
│   └── database_manager.py           # SQLite database operations
│
├── ml_models/
│   ├── __init__.py
│   ├── spark_gbt_forecaster.py       # Time series forecasting (GBT)
│   └── investment_classifier.py      # Classification model
│
├── chatbot/
│   ├── __init__.py
│   ├── investment_chatbot.py         # Basic chatbot template
│   └── ai_prediction_chatbot.py      # Advanced chatbot with ML & graphs
│
├── dashboard/
│   ├── __init__.py
│   └── dashboard_app.py              # Streamlit dashboard
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py              # Unit tests
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### Step 2: Setup Ollama (for LLM)

```bash
# Pull and run Ollama Docker container
docker pull ollama/ollama
docker run -d --name ollama -p 11434:11434 ollama/ollama

# Pull Llama 3.2 model
docker exec -it ollama ollama pull llama3.2

# Verify it's running
docker ps | grep ollama
```

### Step 3: Run the Pipeline

```bash
# Run main pipeline
python main.py

# Follow menu options:
# 1. Data Collection
# 2. Preprocessing
# 3. Database Setup
# 4. Train ML Models
# 5. Run Chatbot
# 6. Run Dashboard
```

---

## 📝 Assignment Tasks

### **Task 1: Data Collection** (20 points)

**File**: `data_collection/stock_downloader.py`

**What to implement**:
1. Connect to Yahoo Finance API using `yfinance`
2. Download 5 years of historical data for: AAPL, MSFT, GOOGL, AMZN, TSLA
3. Save data as CSV files in `data/stock_data/`
4. Include columns: Date, Open, High, Low, Close, Volume

**Expected Output**:
```
data/stock_data/
├── AAPL_stock_data.csv (1254 rows)
├── MSFT_stock_data.csv (1254 rows)
├── GOOGL_stock_data.csv (1254 rows)
├── AMZN_stock_data.csv (1254 rows)
└── TSLA_stock_data.csv (1254 rows)
```

**Hints**:
- Use `yfinance.download(ticker, start_date, end_date)`
- Date range: 2020-01-01 to today
- Handle API errors gracefully

---

### **Task 2: Data Preprocessing with PySpark** (30 points)

**File**: `preprocessing/spark_preprocessor.py`

**What to implement**:
1. Load CSV files into Spark DataFrames
2. Feature Engineering:
   - Moving Averages: MA_7, MA_30, MA_90 (use Window functions)
   - RSI (Relative Strength Index) - 14-day period
   - Volatility (30-day rolling standard deviation)
   - Daily Returns: `(Close - Previous Close) / Previous Close`
   - Sharpe Ratio: `Mean Return / Std Dev Return`
3. Handle missing values (forward fill or drop)
4. Save processed data as Parquet: `data/processed_stocks.parquet`

**Expected Output**:
```
Columns: Ticker, Date, Open, High, Low, Close, Volume,
         MA_7, MA_30, MA_90, RSI, Volatility, Daily_Return, Sharpe_Ratio
Rows: ~6,270 (1254 per ticker × 5 tickers)
```

**Hints**:
- Use `Window.partitionBy('Ticker').orderBy('Date')` for moving averages
- RSI formula: `RSI = 100 - (100 / (1 + RS))` where `RS = Avg Gain / Avg Loss`
- Use `df.withColumn()` for feature creation

---

### **Task 3: SQLite Database** (15 points)

**File**: `sql_interface/database_manager.py`

**What to implement**:
1. Create SQLite database: `data/financial_data.db`
2. Design schema:
```sql
CREATE TABLE stock_data (
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
```
3. Load data from Parquet into SQLite
4. Implement query methods: `get_stock_data(ticker)`, `get_latest_prices()`

**Expected Output**:
- Database with 6,270 rows
- Fast queries using indexes

---

### **Task 4: ML Time Series Forecasting** (35 points)

**File**: `ml_models/spark_gbt_forecaster.py`

**What to implement**:
1. **Feature Engineering**:
   - Create lagged features: `Close_lag_1` to `Close_lag_30` (30 days of history)
   - Create lagged features for: Open, High, Low, Volume (30 days each)
   - Total: 150 features (5 indicators × 30 lags)
   - Use PySpark Window functions with `lag()`

2. **Model Training**:
   - Model: Spark MLlib `GBTRegressor` (Gradient Boosted Trees)
   - Hyperparameters:
     - `maxIter=100` (number of trees)
     - `maxDepth=6`
     - `stepSize=0.1` (learning rate)
     - `subsamplingRate=0.8`
   - Target: Predict `Close` price 7 days into the future

3. **Evaluation**:
   - Split: 80% train, 10% validation, 10% test
   - Metrics: RMSE, MAE, R², Mean % Error

4. **Prediction**:
   - Implement `predict_future(ticker, num_days)` to forecast next N days
   - Add realistic variation (1% std deviation) between predicted days

**Expected Output**:
```
Model Performance:
  Test RMSE: $25-30
  Test R²: 0.94+ (94% accuracy)
  Mean % Error: 5-6%
```

**Hints**:
- Create features in batches (1-10, 11-20, 21-30) to avoid StackOverflowError
- Use `.cache()` and `.count()` after each batch
- Save model: `model.write().overwrite().save(path)`

---

### **Task 5: ML Classification** (20 points)

**File**: `ml_models/investment_classifier.py`

**What to implement**:
1. **Feature Engineering**:
   - Total Return (first to last price)
   - Recent Returns (7-day, 30-day)
   - Average RSI, Current RSI
   - Volatility, Sharpe Ratio
   - Price trends (MA7 vs MA30, MA30 vs MA90)
   - Total: 17 features

2. **Classification**:
   - Model: `RandomForestClassifier` (sklearn or Spark MLlib)
   - Labels: High (score ≥ 7), Medium (4-7), Low (< 4)
   - Composite score formula:
     ```
     Score = (Total_Return × 0.3) + (Trend_Score × 0.2) +
             (RSI_Score × 0.15) + (Volatility_Score × 0.15) +
             (Sharpe_Score × 0.2)
     ```

3. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score

**Expected Output**:
```
Classification Results:
  AAPL: High (Score: 7.8)
  MSFT: High (Score: 7.5)
  GOOGL: Medium (Score: 6.2)
  AMZN: Medium (Score: 5.9)
  TSLA: High (Score: 8.1)
```

---

### **Task 6: AI Chatbot** (30 points)

**File**: `chatbot/ai_prediction_chatbot.py`

**What to implement**:
1. **Database Integration**:
   - Query SQLite for historical data
   - Methods: `get_stock_data(ticker, days)`

2. **ML Integration**:
   - Load Spark GBT model
   - Generate predictions: `get_prediction(ticker, num_days)`

3. **Graph Generation**:
   - Use Matplotlib to create prediction charts
   - Plot historical prices (last 30 days) + predicted prices (next 7 days)
   - Convert to base64 PNG for display

4. **Natural Language Processing**:
   - Intent detection: prediction, data, general chat
   - Extract ticker and number of days from query
   - Examples:
     - "Predict AAPL next 7 days" → prediction intent
     - "Show TSLA data" → data intent
     - "What is RSI?" → chat intent

5. **LLM Integration**:
   - Use Ollama API with Llama 3.2 for general questions
   - Build informative prompts

**Expected Queries**:
```
User: "Predict AAPL next 7 days"
Bot: [Shows text prediction + graph image]

User: "Tell me about TSLA"
Bot: [Shows latest 10 days of data with indicators]

User: "What is moving average?"
Bot: [Llama explains concept]
```

**Hints**:
- Use Streamlit for UI: `st.chat_input()`, `st.chat_message()`
- Matplotlib graph: `plt.savefig(buf, format='png')` → `base64.b64encode()`
- Ollama API: `ollama.chat(model='llama3.2', messages=[...])`

---

### **Task 7: Dashboard** (20 points)

**File**: `dashboard/dashboard_app.py`

**What to implement**:
1. **Tab 1: Stock Data Viewer**
   - Dropdown to select ticker
   - Display historical data table
   - Line chart for closing prices

2. **Tab 2: Technical Indicators**
   - Plot MA7, MA30, MA90 on same chart
   - Display RSI with overbought/oversold zones
   - Show volatility trends

3. **Tab 3: ML Predictions**
   - Load Spark GBT model
   - Input: Select ticker, number of days to predict
   - Output: Prediction table + interactive chart

4. **Tab 4: Investment Classification**
   - Show classification results for all tickers
   - Display scores and explanations

5. **Tab 5: Model Explanations**
   - Explain all 150 features used in forecasting
   - Explain 17 features used in classification
   - Show model architecture and performance metrics

**Expected Output**:
- Interactive dashboard accessible at `http://localhost:8501`
- Professional UI with charts and tables

---

### **Task 8: Testing** (10 points)

**File**: `tests/test_pipeline.py`

**What to implement**:
1. Unit tests for data collection
2. Tests for feature engineering
3. Tests for database operations
4. Tests for ML model predictions
5. Integration test for full pipeline

**Use pytest**:
```bash
pytest tests/test_pipeline.py -v
```

---

## 🔧 Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
```
pyspark==3.5.0
pandas==2.0.3
numpy==1.24.3
yfinance==0.2.28
matplotlib==3.7.2
streamlit==1.28.0
scikit-learn==1.3.0
ollama==0.1.0
sqlite3 (built-in)
pytest==7.4.0
```

### 2. Setup Ollama with Docker

**Step 2.1: Install Docker**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Step 2.2: Run Ollama**
```bash
# Pull Ollama image
docker pull ollama/ollama

# Run Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama

# Verify it's running
docker ps | grep ollama
```

**Step 2.3: Install Llama Model**
```bash
# Pull Llama 3.2 (3B parameters, ~2GB)
docker exec -it ollama ollama pull llama3.2

# Test it
docker exec -it ollama ollama run llama3.2 "Hello, what is AI?"
```

**Alternative: Larger Model (Optional)**
```bash
# Llama 3.1 (8B parameters, better quality but slower)
docker exec -it ollama ollama pull llama3.1
```

**Troubleshooting**:
```bash
# Check Ollama logs
docker logs ollama

# Restart Ollama
docker restart ollama

# Check available models
docker exec -it ollama ollama list
```

### 3. Verify Setup

```bash
# Test Ollama API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is machine learning?"
}'

# Test Python Ollama client
python -c "import ollama; print(ollama.chat(model='llama3.2', messages=[{'role':'user','content':'Hi'}]))"
```

---

## 📊 ML Models Explained

### Model 1: Spark GBT Forecaster (Time Series)

**Algorithm**: Gradient Boosted Trees (GBT)
**Framework**: PySpark MLlib
**Purpose**: Predict stock prices 7 days into the future

**How it works**:
1. **Input Features**: 150 lagged features (30 days × 5 indicators)
   - `Close_lag_1`, `Close_lag_2`, ..., `Close_lag_30`
   - `Open_lag_1`, `Open_lag_2`, ..., `Open_lag_30`
   - `High_lag_1`, ..., `Low_lag_1`, ..., `Volume_lag_1`, ...

2. **Target**: Close price 7 days in the future

3. **Architecture**:
   - 100 decision trees (gradient boosted)
   - Max depth: 6
   - Learning rate: 0.1
   - Subsample ratio: 0.8

4. **Training**:
   - Window function creates sequences
   - Each row = (30 days history) → (future price)
   - Example: `[Day 1-30 prices] → Day 37 price`

5. **Prediction**:
   - Take last 30 days
   - Predict next 7 days iteratively
   - Add realistic variation (1% std dev)

**Why GBT?**
- ✅ Handles non-linear patterns
- ✅ Captures complex feature interactions
- ✅ Robust to outliers
- ✅ Distributed training with PySpark

**Expected Performance**:
- R² Score: 94%+
- RMSE: $25-30
- Mean Error: 5-6%

---

### Model 2: Investment Classifier

**Algorithm**: Random Forest Classifier
**Framework**: Scikit-learn or Spark MLlib
**Purpose**: Classify stocks as High/Medium/Low investment potential

**How it works**:
1. **Input Features**: 17 aggregate features
   - Returns: Total, 7-day, 30-day
   - Technical: RSI, Volatility, Sharpe Ratio
   - Trends: MA comparisons, price momentum

2. **Composite Score** (0-10):
   ```
   Score = (Total_Return × 0.3) +      # 30% weight
           (Trend_Score × 0.2) +        # 20% weight
           (RSI_Score × 0.15) +         # 15% weight
           (Volatility_Score × 0.15) +  # 15% weight
           (Sharpe_Score × 0.2)         # 20% weight
   ```

3. **Classification**:
   - High: Score ≥ 7 (Strong buy signal)
   - Medium: 4 ≤ Score < 7 (Hold/Watch)
   - Low: Score < 4 (Avoid/Sell)

4. **Model**: Random Forest with 100 trees

**Why Random Forest?**
- ✅ Handles mixed feature types
- ✅ Feature importance analysis
- ✅ No feature scaling needed
- ✅ Resistant to overfitting

---

## 🎓 Grading Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Data Collection | 20 | Correct API usage, data quality, error handling |
| Preprocessing | 30 | PySpark usage, feature engineering correctness, data validation |
| Database | 15 | Schema design, data integrity, query efficiency |
| Time Series ML | 35 | Model accuracy (R² ≥ 0.90), feature engineering, predictions |
| Classification | 20 | Classification accuracy, feature design, interpretability |
| Chatbot | 30 | NLP intent detection, ML integration, graph generation, LLM usage |
| Dashboard | 20 | UI quality, interactivity, visualizations |
| Testing | 10 | Test coverage, edge cases |
| Code Quality | 10 | Documentation, style, modularity |
| Documentation | 10 | README, comments, setup guide |
| **Total** | **200** | |

---

## 📚 Resources

### PySpark:
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Window Functions Tutorial](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)

### Machine Learning:
- [Gradient Boosted Trees](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression)
- [Time Series Forecasting](https://www.kaggle.com/learn/time-series)
- [Technical Indicators](https://www.investopedia.com/technical-analysis-4689657)

### LLMs:
- [Ollama Documentation](https://ollama.ai/docs)
- [Llama Models](https://ai.meta.com/llama/)
- [Streamlit Chatbot Tutorial](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)

### Finance:
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [RSI Calculation](https://www.investopedia.com/terms/r/rsi.asp)
- [Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp)

---

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'pyspark'"
```bash
pip install pyspark==3.5.0
```

### Issue 2: "Ollama connection refused"
```bash
# Check if Ollama is running
docker ps | grep ollama

# Start if not running
docker start ollama

# Check logs
docker logs ollama
```

### Issue 3: "StackOverflowError in PySpark"
```python
# Create features in batches and cache
df = df.cache()
df.count()  # Force computation
```

### Issue 4: "SQLite database locked"
```python
# Close connections properly
conn.close()

# Or use context manager
with sqlite3.connect(db_path) as conn:
    # operations
    pass
```

### Issue 5: "Matplotlib backend error"
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

---

## 🏆 Bonus Challenges (Extra Credit)

1. **Advanced NLP** (+10 points): Add LLaVA model for chart image analysis
2. **Real-time Data** (+10 points): Stream live stock prices using WebSocket
3. **Deep Learning** (+15 points): Add LSTM model for comparison
4. **Deployment** (+10 points): Deploy dashboard to cloud (Heroku/AWS)
5. **API** (+10 points): Create FastAPI REST endpoints for predictions

---

## 📝 Submission Guidelines

1. **Code**: Push to GitHub repository
2. **Documentation**: Complete README with:
   - Setup instructions
   - How to run pipeline
   - Model performance metrics
   - Screenshots of dashboard
3. **Video Demo**: 5-minute walkthrough (optional but recommended)
4. **Report**: PDF with:
   - Architecture diagram
   - Feature engineering decisions
   - Model evaluation results
   - Challenges faced and solutions

**Deadline**: [Set by instructor]

**Submission Format**:
```
submission/
├── code/                 # Complete project code
├── README.md            # Project documentation
├── REPORT.pdf           # Technical report
├── demo_video.mp4       # Demo video (optional)
└── screenshots/         # Dashboard screenshots
```

---

## 💡 Tips for Success

1. **Start Early** - Data collection can take time
2. **Test Incrementally** - Don't wait to test everything at once
3. **Use Version Control** - Commit frequently to Git
4. **Document as You Go** - Write comments and docstrings
5. **Ask Questions** - Use course forums or office hours
6. **Validate Data** - Check data quality at each step
7. **Monitor Resources** - PySpark can be memory-intensive
8. **Handle Errors** - Add try-except blocks everywhere

---

## 📞 Support

- **Instructor**: [Your Name]
- **Office Hours**: [Times]
- **Course Forum**: [Link]
- **Email**: [Your Email]

---

## 📄 License

This project template is for educational purposes only. Stock market predictions are for learning - not financial advice!

**Disclaimer**: Past performance does not guarantee future results. This project is for educational purposes only and should not be used for actual investment decisions.

---

**Good luck! 🚀 You're building a real-world data engineering + ML system!**
