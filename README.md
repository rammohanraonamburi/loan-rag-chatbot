# 🏦 Loan Application Chatbot

A **smart Q&A chatbot** for loan application analysis, built with Streamlit and pandas for easy deployment and use.

## 🌐 Live Demo

**[🚀 Try the Live Demo](https://your-demo-link-here.streamlit.app)**

*Note: Replace the link above with your actual deployment URL after deploying the application.*

## 🚀 Features

- **Smart Q&A**: Ask questions about loan applications, approval rates, and patterns
- **Real-time Statistics**: Live dashboard with key metrics
- **Interactive Chat**: Modern Streamlit interface with chat history
- **Data Analysis**: Comprehensive insights from loan application data
- **Easy Deployment**: Lightweight and works on any platform
- **No External APIs**: Works completely offline

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Backend**: Python
- **Deployment**: Streamlit Cloud

## 📊 Dataset

The system uses a comprehensive loan application dataset containing:
- **615 training samples** and **367 test samples**
- Features: Gender, Marital Status, Dependents, Education, Income, Loan Amount, Credit History, Property Area
- Target: Loan Approval Status (Y/N)

## 🏗️ Project Structure

```
rag_loan_chatbot/
├── simple_chatbot.py     # Main chatbot application
├── streamlit_app.py      # Streamlit web application (legacy)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── *.csv                # Loan dataset files
```

## ⚙️ Installation & Setup

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/rag_loan_chatbot

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run simple_chatbot.py
```

The application will be available at `http://localhost:8501`

### 3. Prepare Data

Make sure your loan dataset files are in the project directory:
- `Training Dataset.csv`
- `Test Dataset.csv`

## 🚀 Quick Deployment

### Streamlit Cloud (Recommended)

1. **Fork this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Deploy with these settings:**
   - Repository: `your-username/your-repo-name`
   - Main file path: `rag_loan_chatbot/simple_chatbot.py`
   - Python version: 3.9+

5. **Update the demo link in README.md with your deployment URL**

That's it! No environment variables needed.

## 💬 Usage Examples

### Sample Questions You Can Ask:

1. **General Statistics**
   - "What is the overall loan approval rate?"
   - "How many applications are in the dataset?"

2. **Pattern Analysis**
   - "How does education level affect loan approval?"
   - "What are the income requirements for loan approval?"
   - "How does credit history impact loan decisions?"

3. **Demographic Analysis**
   - "What is the difference between urban, semiurban, and rural property areas?"
   - "How does marital status affect loan approval?"

4. **Specific Queries**
   - "Find similar applications to LP001002"
   - "Show me applications with high income but low approval rates"

## 🎯 Key Features

### 1. **Intelligent Document Retrieval**
- Semantic search through loan applications
- Context-aware document ranking
- Metadata-based filtering

### 2. **Interactive Chat Interface**
- Real-time conversation
- Message history
- Retrieved document inspection
- Sample query suggestions

### 3. **Data Visualization**
- Income distribution charts
- Education level analysis
- Property area demographics
- Approval rate statistics

### 4. **Advanced Analytics**
- Similar application finding
- Pattern recognition
- Statistical analysis
- Trend identification

## 🔧 Configuration Options

### Model Selection
- OpenAI GPT-3.5-turbo (default)
- OpenAI GPT-4
- Local models (coming soon)

### Retrieval Settings
- Number of documents to retrieve (1-10)
- Similarity thresholds
- Metadata filters

## 📈 Performance

- **Response Time**: < 3 seconds for most queries
- **Accuracy**: High relevance through semantic search
- **Scalability**: Handles thousands of loan applications
- **Memory Efficient**: Optimized vector storage

## 🚀 Advanced Features

### 1. **Similar Application Analysis**
Find applications similar to a specific loan ID:
```python
similar_apps = chatbot.find_similar_applications("LP001002", n_results=5)
```

### 2. **Metadata-Based Search**
Search by specific criteria:
```python
urban_apps = vector_store.search_by_metadata({"property_area": "Urban"})
```

### 3. **Statistical Analysis**
Get comprehensive dataset statistics:
```python
stats = data_processor.get_statistics()
```

## 🔒 Security & Privacy

- No sensitive data is stored permanently
- API keys are managed securely through environment variables
- Local vector database for data privacy
- No external data transmission beyond OpenAI API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **ChromaDB** for vector database
- **Streamlit** for web interface
- **LangChain** for RAG framework
- **Sentence Transformers** for embeddings

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description

---

**Built with ❤️ for intelligent loan application analysis** 