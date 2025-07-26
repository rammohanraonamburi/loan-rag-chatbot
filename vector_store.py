import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import os

class LoanVectorStore:
    def __init__(self, collection_name: str = "loan_applications", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store for loan applications.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Loan application documents for RAG system"}
            )
            print(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'id', 'content', and 'metadata' keys
        """
        if not documents:
            print("No documents to add")
            return
            
        # Prepare data for ChromaDB
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Add documents to collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional filters for metadata
            
        Returns:
            List of relevant documents with scores
        """
        try:
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            n_results: Number of results to return
            
        Returns:
            List of documents matching the metadata filters
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results
            )
            
            formatted_results = []
            for i in range(len(results['documents'])):
                result = {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during metadata search: {e}")
            return []
    
    def get_similar_applications(self, loan_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar loan applications based on a specific loan ID.
        
        Args:
            loan_id: ID of the loan application to find similar ones for
            n_results: Number of similar applications to return
            
        Returns:
            List of similar loan applications
        """
        try:
            # First, get the document for the given loan ID
            loan_doc = self.collection.get(
                where={"loan_id": loan_id},
                limit=1
            )
            
            if not loan_doc['documents']:
                print(f"No document found for loan ID: {loan_id}")
                return []
            
            # Use the content to find similar documents
            similar_results = self.collection.query(
                query_texts=[loan_doc['documents'][0]],
                n_results=n_results + 1,  # +1 to exclude the original
                where={"loan_id": {"$ne": loan_id}}  # Exclude the original loan
            )
            
            formatted_results = []
            for i in range(len(similar_results['documents'][0])):
                result = {
                    'content': similar_results['documents'][0][i],
                    'metadata': similar_results['metadatas'][0][i],
                    'id': similar_results['ids'][0][i],
                    'similarity_score': 1 - similar_results['distances'][0][i] if 'distances' in similar_results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error finding similar applications: {e}")
            return [] 