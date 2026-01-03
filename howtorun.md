# How to Run the AI Financial Analysis Platform

This guide provides step-by-step instructions to set up and run the project.

## 1. Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **Docker** (for Ollama)
- **Ollama** (for LLM capabilities)

## 2. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Model Training (Required)

Before running the dashboard or chatbot, you must train the machine learning models. The Gradient Boosted Trees (GBT) model is required for predictions.

```bash
# Train the GBT Forecaster
python ml_models/spark_gbt_forecaster.py
```
*Note: This step may take a few minutes as it involves Spark feature engineering and training.*

## 4. Running the Dashboard

The dashboard is built with Streamlit and provides a visual interface for stock data, indicators, and predictions.

```bash
streamlit run dashboard/dashboard_app.py
```
Access the dashboard at: `http://localhost:8501`

## 5. Running the Chatbot

The chatbot uses Ollama (Llama 3.2) to answer questions and provide insights.

1.  **Start Ollama**:
    ```bash
    docker start ollama
    # Or run if not created:
    # docker run -d --name ollama -p 11434:11434 ollama/ollama
    ```

2.  **Run the Chatbot**:
    ```bash
    streamlit run chatbot/ai_prediction_chatbot.py
    ```
Access the chatbot at: `http://localhost:8501` (or the port shown in terminal, usually 8502 if dashboard is running).

## 6. Running the Full Pipeline (Optional)

You can also use the main orchestrator script to access a menu of all options:

```bash
python main.py
```

## Troubleshooting

-   **Model Loading Error**: If you see an error about `Pipeline` vs `PipelineModel`, ensure you have the latest code in `ml_models/spark_gbt_forecaster.py` which uses `PipelineModel.load()`.
-   **ModuleNotFoundError (e.g., No module named 'pyspark')**: This means you haven't activated the virtual environment. Run `source venv/bin/activate` before running any python scripts.
-   **Ollama Connection**: Ensure the Docker container for Ollama is running (`docker ps`).
