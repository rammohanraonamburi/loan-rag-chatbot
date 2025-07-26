# 🚀 Deployment Guide for Loan Application RAG Chatbot

This guide provides multiple deployment options for the RAG chatbot.

## 📋 Prerequisites

- Python 3.8+
- Git repository with your code
- OpenAI API key (optional - system works with fallback responses)

## 🎯 Option 1: Streamlit Cloud (Recommended)

### **Step 1: Prepare Your Repository**

1. **Ensure your repository structure:**
   ```
   rag_loan_chatbot/
   ├── streamlit_app.py
   ├── data_processor.py
   ├── vector_store.py
   ├── rag_chatbot.py
   ├── requirements.txt
   ├── README.md
   └── .streamlit/
       └── config.toml
   ```

2. **Add data files to your repository:**
   - Copy your CSV files to the repository
   - Update the data paths in `streamlit_app.py`

### **Step 2: Deploy to Streamlit Cloud**

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app:**
   - **Repository**: `your-username/your-repo-name`
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `rag_loan_chatbot/streamlit_app.py`
   - **App URL**: Choose a custom subdomain (optional)

5. **Click "Deploy"**

### **Step 3: Configure Environment Variables**

1. **In Streamlit Cloud dashboard, go to your app settings**
2. **Add environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)

### **Step 4: Update Data Paths**

If your data files are in the repository, update the paths in `streamlit_app.py`:

```python
# Change from:
train_path="../archive/Training Dataset.csv"
test_path="../archive/Test Dataset.csv"

# To:
train_path="Training Dataset.csv"
test_path="Test Dataset.csv"
```

## 🌐 Option 2: Heroku

### **Step 1: Create Heroku App**

```bash
# Install Heroku CLI
# Create app
heroku create your-rag-chatbot

# Add buildpacks
heroku buildpacks:add heroku/python
```

### **Step 2: Create Procfile**

Create a `Procfile` in your root directory:
```
web: streamlit run rag_loan_chatbot/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### **Step 3: Deploy**

```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## ☁️ Option 3: Google Cloud Platform

### **Step 1: Create App Engine App**

1. **Create `app.yaml`:**
```yaml
runtime: python39
entrypoint: streamlit run rag_loan_chatbot/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0

env_variables:
  OPENAI_API_KEY: "your-api-key-here"
```

2. **Deploy:**
```bash
gcloud app deploy
```

## 🐳 Option 4: Docker

### **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "rag_loan_chatbot/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Step 2: Build and Run**

```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

## 🔧 Environment Variables

Set these environment variables in your deployment platform:

- `OPENAI_API_KEY`: Your OpenAI API key (optional - system works without it)

## 📊 Data Management

### **Option A: Include Data in Repository**
- Add CSV files to your repository
- Update file paths in the code

### **Option B: Use External Storage**
- Upload data to cloud storage (AWS S3, Google Cloud Storage)
- Update code to download data on startup

### **Option C: Use Database**
- Set up a database (PostgreSQL, MongoDB)
- Update data loading code

## 🚨 Troubleshooting

### **Common Issues:**

1. **Import Errors:**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **Data Path Issues:**
   - Verify file paths are correct for deployment environment
   - Use absolute paths or environment variables

3. **Memory Issues:**
   - Reduce number of documents retrieved
   - Use smaller embedding models

4. **API Key Issues:**
   - System works without OpenAI API key
   - Check environment variable configuration

## 📈 Performance Optimization

1. **Reduce Model Size:**
   - Use smaller sentence transformer models
   - Limit number of retrieved documents

2. **Caching:**
   - Enable Streamlit caching for expensive operations
   - Cache vector embeddings

3. **Database Optimization:**
   - Use persistent vector database
   - Index frequently queried fields

## 🔒 Security Considerations

1. **API Keys:**
   - Never commit API keys to repository
   - Use environment variables

2. **Data Privacy:**
   - Ensure no sensitive data in logs
   - Use secure data storage

3. **Access Control:**
   - Implement authentication if needed
   - Rate limiting for API calls

## 📞 Support

For deployment issues:
1. Check the deployment platform's documentation
2. Review error logs
3. Test locally first
4. Use the fallback response system if API issues occur 