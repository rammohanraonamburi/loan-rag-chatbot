#!/usr/bin/env python3
"""
Setup script for the Loan Application RAG Chatbot
This script initializes the system, processes data, and sets up the vector database.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'chromadb', 
        'sentence_transformers', 'langchain', 'openai', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_data_files():
    """Check if the required data files exist."""
    data_path = Path("../archive")
    required_files = ["Training Dataset.csv", "Test Dataset.csv"]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nPlease place the data files in: {data_path.absolute()}")
        return False
    
    print("✅ All data files found")
    return True

def check_environment():
    """Check environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("⚠️  OPENAI_API_KEY not found or not set properly")
        print("   You can set it in a .env file or export it:")
        print("   echo 'OPENAI_API_KEY=your_key_here' > .env")
        return False
    
    print("✅ OpenAI API key found")
    return True

def initialize_system():
    """Initialize the RAG system."""
    try:
        print("\n🚀 Initializing RAG system...")
        
        # Import modules
        from data_processor import LoanDataProcessor
        from vector_store import LoanVectorStore
        from rag_chatbot import LoanRAGChatbot
        
        # Initialize data processor
        print("📊 Loading and processing data...")
        data_processor = LoanDataProcessor(
            train_path="../archive/Training Dataset.csv",
            test_path="../archive/Test Dataset.csv"
        )
        
        data_processor.load_data()
        data_processor.clean_data()
        
        # Create documents
        print("📝 Creating documents for vector store...")
        documents = data_processor.create_documents()
        print(f"   Created {len(documents)} documents")
        
        # Initialize vector store
        print("🗄️  Setting up vector database...")
        vector_store = LoanVectorStore()
        
        # Add documents to vector store
        print("💾 Adding documents to vector store...")
        vector_store.add_documents(documents)
        
        # Test vector store
        stats = vector_store.get_collection_stats()
        print(f"   Vector store contains {stats['total_documents']} documents")
        
        # Initialize chatbot
        print("🤖 Initializing chatbot...")
        chatbot = LoanRAGChatbot(vector_store, use_openai=True)
        
        # Test the system
        print("🧪 Testing the system...")
        test_query = "What is the overall loan approval rate?"
        response = chatbot.chat(test_query)
        print(f"   Test query: '{test_query}'")
        print(f"   Response: {response['response'][:100]}...")
        
        print("\n✅ System initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {str(e)}")
        return False

def main():
    """Main setup function."""
    print("🏦 Loan Application RAG Chatbot Setup")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    print("\n2. Checking data files...")
    if not check_data_files():
        sys.exit(1)
    
    # Check environment
    print("\n3. Checking environment...")
    if not check_environment():
        print("   Continuing without API key (some features may not work)")
    
    # Initialize system
    print("\n4. Initializing system...")
    if not initialize_system():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("   streamlit run streamlit_app.py")
    print("\nThe application will be available at: http://localhost:8501")

if __name__ == "__main__":
    main() 