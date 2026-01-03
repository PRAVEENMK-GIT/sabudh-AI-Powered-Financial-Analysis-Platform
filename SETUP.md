# Complete Setup Guide - AI Financial Analysis Platform

This guide will help you set up everything from scratch, including Docker, Ollama, and LLM models.

---

## 📋 System Requirements

### Minimum Requirements:
- **OS**: Ubuntu 20.04+, macOS 12+, or Windows 10+ (WSL2)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher
- **Java**: 8 or 11 (for PySpark)

### Check Your System:
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check Java version
java -version  # Should be 8 or 11

# Check available RAM
free -h  # Linux
vm_stat | grep "Pages free" | awk '{print $3 * 4096 / 1024 / 1024 " MB"}'  # macOS
```

---

## 🐳 Step 1: Install Docker

Docker is required to run Ollama (the LLM runtime).

### Ubuntu/Debian:
```bash
# Update package index
sudo apt update

# Install dependencies
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Test Docker
docker --version
docker run hello-world
```

### macOS:
```bash
# Install Docker Desktop from: https://www.docker.com/products/docker-desktop/
# Or use Homebrew:
brew install --cask docker

# Start Docker Desktop app
# Verify installation:
docker --version
```

### Windows (WSL2):
1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
3. Enable WSL2 backend in Docker Desktop settings
4. Verify:
```powershell
docker --version
```

---

## 🤖 Step 2: Install Ollama (LLM Runtime)

Ollama runs large language models locally in a Docker container.

### Pull Ollama Docker Image:
```bash
# Pull the latest Ollama image (~ 500MB)
docker pull ollama/ollama

# Verify
docker images | grep ollama
```

### Run Ollama Container:
```bash
# Start Ollama server in detached mode
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  --restart always \
  ollama/ollama

# Check if it's running
docker ps | grep ollama

# Check logs
docker logs ollama
```

**Expected Output:**
```
CONTAINER ID   IMAGE            STATUS        PORTS
abc123def456   ollama/ollama    Up 2 minutes  0.0.0.0:11434->11434/tcp
```

### Test Ollama API:
```bash
# Test API endpoint
curl http://localhost:11434/api/version

# Expected response:
# {"version":"0.1.17"}
```

---

## 🦙 Step 3: Install Llama Models

Ollama supports multiple LLM models. We'll use Llama 3.2 (recommended for students).

### Option 1: Llama 3.2 (Recommended for Students)
**Size**: ~2GB
**Parameters**: 3B
**Speed**: Fast
**Quality**: Good for most tasks

```bash
# Pull Llama 3.2 model
docker exec -it ollama ollama pull llama3.2

# This will download ~2GB, may take 5-10 minutes
```

**Download Progress:**
```
pulling manifest
pulling 4ba6bc3f7d45...  ████████████████  100%
pulling ca54a6fd0c97...  ████████████████  100%
success
```

### Option 2: Llama 3.1 (Better Quality, Slower)
**Size**: ~4.7GB
**Parameters**: 8B
**Speed**: Medium
**Quality**: Better responses

```bash
# Pull Llama 3.1 model (optional)
docker exec -it ollama ollama pull llama3.1
```

### Option 3: Llama 2 (Older, Stable)
```bash
# Pull Llama 2 model (optional)
docker exec -it ollama ollama pull llama2
```

### List Available Models:
```bash
# See all downloaded models
docker exec -it ollama ollama list

# Expected output:
# NAME          ID            SIZE      MODIFIED
# llama3.2:latest  abc123def   2.0GB     2 minutes ago
```

---

## 🧪 Step 4: Test LLM

### Test via Command Line:
```bash
# Run interactive chat
docker exec -it ollama ollama run llama3.2

# Try asking:
# >>> What is machine learning?
# >>> Explain stock market
# >>> /bye  (to exit)
```

### Test via API:
```bash
# Test generation API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Explain artificial intelligence in one sentence.",
  "stream": false
}'

# Test chat API
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is Python?"}
  ]
}'
```

### Test via Python:
```bash
# Install Ollama Python client
pip install ollama

# Test in Python
python3 -c "import ollama; print(ollama.chat(model='llama3.2', messages=[{'role':'user','content':'Hello!'}]))"
```

**Expected Output:**
```python
{
  'model': 'llama3.2',
  'message': {'role': 'assistant', 'content': 'Hello! How can I help you today?'},
  ...
}
```

---

## 🐍 Step 5: Install Python Dependencies

### Create Virtual Environment (Recommended):
```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

### Install Required Packages:
```bash
# Install all dependencies
pip install -r requirements.txt

# This will install:
# - PySpark 3.5.0
# - pandas, numpy
# - yfinance
# - scikit-learn
# - matplotlib, seaborn
# - streamlit
# - ollama
# - pytest
```

### Verify Installation:
```bash
# Test PySpark
python3 -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.appName('test').getOrCreate(); print('✅ PySpark works!')"

# Test yfinance
python3 -c "import yfinance as yf; print('✅ yfinance works!')"

# Test Ollama client
python3 -c "import ollama; print('✅ Ollama client works!')"
```

---

## ☕ Step 6: Install Java (for PySpark)

PySpark requires Java 8 or 11.

### Ubuntu/Debian:
```bash
# Install OpenJDK 11
sudo apt update
sudo apt install -y openjdk-11-jdk

# Verify
java -version
# Should show: openjdk version "11.x.x"
```

### macOS:
```bash
# Install via Homebrew
brew install openjdk@11

# Set JAVA_HOME
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 11)' >> ~/.zshrc
source ~/.zshrc

# Verify
java -version
```

### Set JAVA_HOME (if needed):
```bash
# Find Java path
which java
# or
readlink -f $(which java)

# Set JAVA_HOME in ~/.bashrc or ~/.zshrc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Reload
source ~/.bashrc  # or source ~/.zshrc
```

---

## 🚀 Step 7: Run the Project

### Initialize Project:
```bash
# Navigate to project directory
cd project_template/

# Test configuration
python3 config/config.py

# Should print:
# ✅ Ollama is running
# ✅ Configuration validated!
```

### Run Data Collection:
```bash
# Download stock data
python3 data_collection/stock_downloader.py

# Expected output:
# ✅ Downloaded 1254 rows for AAPL
# ✅ Downloaded 1254 rows for MSFT
# ...
```

### Run Preprocessing:
```bash
# Process data with PySpark
python3 preprocessing/spark_preprocessor.py

# Expected output:
# ✅ Created MA_7, MA_30, MA_90
# ✅ Created RSI column
# ✅ Saved processed_stocks.parquet
```

### Train ML Model:
```bash
# Train GBT forecaster
python3 ml_models/spark_gbt_forecaster.py

# Expected output:
# Test R²: 0.94
# Test RMSE: $25.32
```

### Run Chatbot:
```bash
# Start chatbot (Streamlit)
streamlit run chatbot/ai_prediction_chatbot.py --server.port 8502

# Open browser: http://localhost:8502
```

### Run Dashboard:
```bash
# Start dashboard (Streamlit)
streamlit run dashboard/dashboard_app.py --server.port 8501

# Open browser: http://localhost:8501
```

---

## 🐛 Troubleshooting

### Issue 1: "Docker command not found"
```bash
# Check if Docker is installed
which docker

# If not found, install Docker (see Step 1)

# Check if Docker service is running
sudo systemctl status docker
sudo systemctl start docker
```

### Issue 2: "Permission denied while connecting to Docker"
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, OR
newgrp docker

# Test
docker ps
```

### Issue 3: "Ollama connection refused"
```bash
# Check if Ollama container is running
docker ps | grep ollama

# If not running, start it
docker start ollama

# Check logs
docker logs ollama

# Check if port 11434 is accessible
curl http://localhost:11434/api/version
```

### Issue 4: "Model not found: llama3.2"
```bash
# List available models
docker exec -it ollama ollama list

# If llama3.2 is missing, pull it
docker exec -it ollama ollama pull llama3.2
```

### Issue 5: "Java not found" or "JAVA_HOME not set"
```bash
# Find Java
which java
readlink -f $(which java)

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Add to ~/.bashrc permanently
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

### Issue 6: "ModuleNotFoundError: No module named 'pyspark'"
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify
python3 -c "import pyspark; print(pyspark.__version__)"
```

### Issue 7: "StackOverflowError in PySpark"
This happens when creating too many lagged features at once.

**Solution**: Create features in batches with `.cache()`:
```python
# Create lag 1-10
for i in range(1, 11):
    df = df.withColumn(f'lag_{i}', lag(col('Close'), i).over(window))
df = df.cache()
df.count()  # Force computation

# Create lag 11-20
# ... repeat
```

### Issue 8: "Streamlit command not found"
```bash
# Install streamlit
pip install streamlit

# Verify
streamlit --version
```

---

## 🔧 Useful Docker Commands

```bash
# Start Ollama
docker start ollama

# Stop Ollama
docker stop ollama

# Restart Ollama
docker restart ollama

# View Ollama logs
docker logs ollama
docker logs -f ollama  # Follow logs

# Remove Ollama container
docker stop ollama
docker rm ollama

# Remove Ollama image
docker rmi ollama/ollama

# Check disk usage
docker system df

# Clean up unused Docker resources
docker system prune -a
```

---

## 📊 Resource Monitoring

### Check RAM Usage:
```bash
# Linux
free -h

# macOS
vm_stat
```

### Check Docker Resource Usage:
```bash
# View container stats
docker stats ollama

# Expected output:
# NAME    CPU %  MEM USAGE / LIMIT  MEM %   NET I/O
# ollama  2.5%   1.2GiB / 8GiB      15%     0B / 0B
```

### PySpark Memory Settings:
If you have limited RAM, reduce Spark memory in `config/config.py`:
```python
# For 8GB RAM systems
SPARK_DRIVER_MEMORY = "2g"
SPARK_EXECUTOR_MEMORY = "2g"

# For 16GB RAM systems
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"
```

---

## 🎓 Next Steps

1. ✅ Complete all setup steps above
2. ✅ Test each component individually
3. ✅ Start implementing student tasks (see README.md)
4. ✅ Run the complete pipeline
5. ✅ Submit your assignment

---

## 📞 Getting Help

- **Ollama Documentation**: https://ollama.ai/docs
- **PySpark Documentation**: https://spark.apache.org/docs/latest/api/python/
- **Docker Documentation**: https://docs.docker.com/
- **Streamlit Documentation**: https://docs.streamlit.io/

---

**Good luck with your project! 🚀**
