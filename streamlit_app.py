import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import LoanDataProcessor
from vector_store import LoanVectorStore
from rag_chatbot import LoanRAGChatbot

# Page configuration
st.set_page_config(
    page_title="Loan Application RAG Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with data processing and vector store."""
    try:
        # Initialize data processor
        data_processor = LoanDataProcessor(
            train_path="Training Dataset.csv",
            test_path="Test Dataset.csv"
        )
        
        # Load and process data
        data_processor.load_data()
        data_processor.clean_data()
        
        # Create documents for vector store
        documents = data_processor.create_documents()
        
        # Initialize vector store
        vector_store = LoanVectorStore()
        
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        # Initialize chatbot
        chatbot = LoanRAGChatbot(vector_store, use_openai=True)
        
        return data_processor, vector_store, chatbot
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None

def display_dataset_overview(data_processor):
    """Display dataset overview and statistics."""
    st.subheader("📊 Dataset Overview")
    
    # Get statistics
    stats = data_processor.get_statistics()
    
    # Create columns for stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", stats['total_applications'])
    
    with col2:
        st.metric("Training Samples", stats['training_samples'])
    
    with col3:
        st.metric("Test Samples", stats['test_samples'])
    
    with col4:
        st.metric("Approval Rate", stats['approval_rate'])
    
    # Income distribution
    st.subheader("💰 Income Distribution")
    income_stats = stats['income_stats']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Income Statistics:**")
        st.write(f"- Mean Income: ${income_stats['mean']:,.2f}")
        st.write(f"- Median Income: ${income_stats['median']:,.2f}")
        st.write(f"- Min Income: ${income_stats['min']:,.2f}")
        st.write(f"- Max Income: ${income_stats['max']:,.2f}")
    
    with col2:
        # Create income distribution plot
        fig = px.histogram(
            data_processor.train_data, 
            x='ApplicantIncome',
            nbins=30,
            title="Applicant Income Distribution"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Education and Property Area distribution
    st.subheader("📈 Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Education distribution
        edu_data = pd.DataFrame(list(stats['education_distribution'].items()), 
                              columns=['Education', 'Count'])
        fig = px.pie(edu_data, values='Count', names='Education', 
                    title="Education Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Property area distribution
        prop_data = pd.DataFrame(list(stats['property_area_distribution'].items()), 
                               columns=['Property Area', 'Count'])
        fig = px.bar(prop_data, x='Property Area', y='Count', 
                    title="Property Area Distribution")
        st.plotly_chart(fig, use_container_width=True)

def display_chat_interface(chatbot):
    """Display the chat interface."""
    st.subheader("💬 Chat with Loan Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about loan applications, approval rates, or any loan-related questions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.chat(prompt)
                st.markdown(response["response"])
                
                # Show retrieved documents (collapsible)
                with st.expander(f"📄 Retrieved Documents ({response['num_docs_retrieved']} docs)"):
                    for i, doc in enumerate(response["relevant_documents"]):
                        st.write(f"**Document {i+1}:**")
                        st.write(f"ID: {doc['id']}")
                        st.write(f"Metadata: {doc['metadata']}")
                        st.write(f"Content: {doc['content'][:200]}...")
                        st.divider()
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["response"]})

def display_sample_queries(chatbot):
    """Display sample queries that users can try."""
    st.subheader("💡 Sample Questions You Can Ask")
    
    sample_queries = [
        "What is the overall loan approval rate?",
        "How does education level affect loan approval?",
        "What are the income requirements for loan approval?",
        "How does credit history impact loan decisions?",
        "What is the difference between urban, semiurban, and rural property areas?",
        "Show me applications with high income but low approval rates",
        "What are the common characteristics of approved loans?",
        "How does marital status affect loan approval?",
        "What is the average loan amount for different education levels?",
        "Find similar applications to LP001002"
    ]
    
    # Create columns for better layout
    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(f"❓ {query}", key=f"query_{i}"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chatbot.chat(query)
                        st.markdown(response["response"])

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Application RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Q&A System for Loan Application Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Model selection
        model_option = st.selectbox(
            "Choose Model",
            ["OpenAI GPT-3.5-turbo", "OpenAI GPT-4", "Local Model (Coming Soon)"],
            index=0
        )
        
        # Number of documents to retrieve
        n_docs = st.slider("Number of documents to retrieve", 1, 10, 5)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # System info
        st.header("ℹ️ System Info")
        st.write("This RAG chatbot uses:")
        st.write("- **Vector Database**: ChromaDB")
        st.write("- **Embeddings**: Sentence Transformers")
        st.write("- **LLM**: OpenAI GPT models")
        st.write("- **Data**: Loan Application Dataset")
    
    # Initialize system
    with st.spinner("Initializing RAG system..."):
        data_processor, vector_store, chatbot = initialize_system()
    
    if data_processor is None:
        st.error("Failed to initialize the system. Please check your data files and API keys.")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dataset Overview", "🔍 Sample Queries"])
    
    with tab1:
        display_chat_interface(chatbot)
    
    with tab2:
        display_dataset_overview(data_processor)
    
    with tab3:
        display_sample_queries(chatbot)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit, LangChain, ChromaDB, and OpenAI</p>
        <p>RAG (Retrieval-Augmented Generation) System for Loan Application Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 