import os
from typing import List, Dict, Any, Optional
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import openai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class LoanRAGChatbot:
    def __init__(self, vector_store, use_openai: bool = True, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the RAG chatbot for loan applications.
        
        Args:
            vector_store: Vector store instance for document retrieval
            use_openai: Whether to use OpenAI API or local model
            model_name: Name of the model to use
        """
        self.vector_store = vector_store
        self.use_openai = use_openai
        self.model_name = model_name
        self.conversation_history = []
        
        # Initialize LLM
        if use_openai:
            self._setup_openai()
        else:
            self._setup_local_model()
            
        # Define system prompt
        self.system_prompt = """You are an intelligent loan application assistant. You have access to a database of loan applications and can provide detailed insights about loan approval patterns, requirements, and statistics.

Your capabilities include:
1. Analyzing loan application data and providing insights
2. Answering questions about loan approval rates and requirements
3. Explaining factors that influence loan decisions
4. Providing statistics about the loan dataset
5. Finding similar loan applications

Always provide accurate, helpful, and detailed responses based on the retrieved information. If you don't have enough information to answer a question, say so clearly."""
        
        # Define prompt template for RAG
        self.rag_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Context information about loan applications:
{context}

Question: {question}

Please provide a comprehensive answer based on the context information above. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""
        )
    
    def _setup_openai(self):
        """Setup OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("Warning: OPENAI_API_KEY not found or invalid")
            print("Using fallback response system based on document analysis")
            self.use_openai = False
            self.llm = None
            return
            
        try:
            openai.api_key = api_key
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1000
            )
            print(f"Initialized OpenAI model: {self.model_name}")
        except Exception as e:
            print(f"Error setting up OpenAI: {e}")
            print("Using fallback response system")
            self.use_openai = False
            self.llm = None
    
    def _setup_local_model(self):
        """Setup local model (placeholder for future implementation)."""
        print("Local model setup not implemented yet. Using fallback response system.")
        self.llm = None
    
    def retrieve_relevant_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: User query
            n_results: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.search(query, n_results=n_results)
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate response using the LLM with retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated response
        """
        if not self.llm:
            return self._generate_fallback_response(query, context_docs)
        
        # Prepare context from retrieved documents
        context_parts = []
        for doc in context_docs:
            context_parts.append(f"Document ID: {doc['id']}\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = self.rag_prompt_template.format(
            context=context,
            question=query
        )
        
        try:
            # Generate response
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            return self._generate_fallback_response(query, context_docs)
    
    def _generate_fallback_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate intelligent responses without using an LLM.
        
        Args:
            query: User query
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated response based on document analysis
        """
        if not context_docs:
            return "I don't have enough information to answer that question. Please try asking about loan applications, approval rates, or specific loan characteristics."
        
        # Analyze the query and documents to generate a response
        query_lower = query.lower()
        
        # Extract key information from documents
        loan_statuses = []
        incomes = []
        education_levels = []
        property_areas = []
        credit_histories = []
        
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            
            # Extract loan status
            if 'loan_status' in metadata:
                loan_statuses.append(metadata['loan_status'])
            
            # Extract income
            if 'applicant_income' in metadata:
                try:
                    income = float(metadata['applicant_income'])
                    incomes.append(income)
                except:
                    pass
            
            # Extract education
            if 'education' in metadata:
                education_levels.append(metadata['education'])
            
            # Extract property area
            if 'property_area' in metadata:
                property_areas.append(metadata['property_area'])
            
            # Extract credit history
            if 'credit_history' in metadata:
                credit_histories.append(metadata['credit_history'])
        
        # Generate responses based on query type
        if 'approval rate' in query_lower or 'approval' in query_lower:
            if loan_statuses:
                approved = sum(1 for status in loan_statuses if status == 'Y')
                total = len(loan_statuses)
                approval_rate = (approved / total) * 100
                return f"Based on the {total} loan applications I analyzed, the approval rate is approximately {approval_rate:.1f}% ({approved} approved out of {total} total)."
            else:
                return "I can see loan applications in the data, but I need to analyze more documents to provide an accurate approval rate."
        
        elif 'income' in query_lower:
            if incomes:
                avg_income = sum(incomes) / len(incomes)
                min_income = min(incomes)
                max_income = max(incomes)
                return f"Based on the analyzed applications, the average applicant income is ${avg_income:,.2f}. Income ranges from ${min_income:,.2f} to ${max_income:,.2f}."
            else:
                return "I can see loan applications with income information, but I need to analyze more documents to provide income statistics."
        
        elif 'education' in query_lower:
            if education_levels:
                edu_counts = {}
                for edu in education_levels:
                    edu_counts[edu] = edu_counts.get(edu, 0) + 1
                
                response = "Education levels in the analyzed applications:\n"
                for edu, count in edu_counts.items():
                    response += f"- {edu}: {count} applications\n"
                return response
            else:
                return "I can see loan applications with education information, but I need to analyze more documents to provide education statistics."
        
        elif 'credit history' in query_lower or 'credit' in query_lower:
            if credit_histories:
                good_credit = sum(1 for credit in credit_histories if credit == '1')
                total_credit = len(credit_histories)
                good_credit_rate = (good_credit / total_credit) * 100
                return f"Based on the analyzed applications, {good_credit_rate:.1f}% of applicants have good credit history ({good_credit} out of {total_credit})."
            else:
                return "I can see loan applications with credit history information, but I need to analyze more documents to provide credit statistics."
        
        elif 'property area' in query_lower or 'urban' in query_lower or 'rural' in query_lower:
            if property_areas:
                area_counts = {}
                for area in property_areas:
                    area_counts[area] = area_counts.get(area, 0) + 1
                
                response = "Property areas in the analyzed applications:\n"
                for area, count in area_counts.items():
                    response += f"- {area}: {count} applications\n"
                return response
            else:
                return "I can see loan applications with property area information, but I need to analyze more documents to provide area statistics."
        
        elif 'similar' in query_lower or 'find' in query_lower:
            if len(context_docs) > 1:
                return f"I found {len(context_docs)} similar loan applications based on your query. These applications share similar characteristics in terms of income, education, property area, and other factors."
            else:
                return "I found one relevant loan application that matches your criteria."
        
        else:
            # Generic response based on available data
            response = f"Based on analyzing {len(context_docs)} relevant loan applications, I can provide the following insights:\n\n"
            
            if loan_statuses:
                approved = sum(1 for status in loan_statuses if status == 'Y')
                total = len(loan_statuses)
                response += f"• Approval rate: {approved}/{total} applications approved\n"
            
            if incomes:
                avg_income = sum(incomes) / len(incomes)
                response += f"• Average income: ${avg_income:,.2f}\n"
            
            if education_levels:
                edu_counts = {}
                for edu in education_levels:
                    edu_counts[edu] = edu_counts.get(edu, 0) + 1
                most_common_edu = max(edu_counts, key=edu_counts.get)
                response += f"• Most common education level: {most_common_edu}\n"
            
            response += "\nFor more specific information, please ask about approval rates, income requirements, education levels, credit history, or property areas."
            
            return response
    
    def chat(self, query: str, n_retrieve: int = 5) -> Dict[str, Any]:
        """
        Main chat method that combines retrieval and generation.
        
        Args:
            query: User query
            n_retrieve: Number of documents to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query, n_retrieve)
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "relevant_documents": relevant_docs,
            "query": query,
            "num_docs_retrieved": len(relevant_docs)
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = self.vector_store.get_collection_stats()
        
        # Add some sample queries and their expected responses
        sample_queries = [
            "What is the overall loan approval rate?",
            "How does education level affect loan approval?",
            "What are the income requirements for loan approval?",
            "How does credit history impact loan decisions?",
            "What is the difference between urban, semiurban, and rural property areas?"
        ]
        
        stats["sample_queries"] = sample_queries
        return stats
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def answer_specific_question(self, question: str, context_type: str = "general") -> str:
        """
        Answer specific types of questions with tailored responses.
        
        Args:
            question: Specific question to answer
            context_type: Type of context to use ("general", "statistics", "patterns")
            
        Returns:
            Tailored response
        """
        if context_type == "statistics":
            # Use metadata-based search for statistical questions
            relevant_docs = self.vector_store.search_by_metadata({}, n_results=10)
        else:
            # Use general semantic search
            relevant_docs = self.retrieve_relevant_documents(question, n_results=5)
        
        return self.generate_response(question, relevant_docs)
    
    def find_similar_applications(self, loan_id: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Find similar loan applications and provide analysis.
        
        Args:
            loan_id: ID of the loan application
            n_results: Number of similar applications to find
            
        Returns:
            Analysis of similar applications
        """
        similar_docs = self.vector_store.get_similar_applications(loan_id, n_results)
        
        if not similar_docs:
            return {
                "response": f"No similar applications found for loan ID: {loan_id}",
                "similar_applications": []
            }
        
        # Generate analysis
        analysis_prompt = f"""
        Analyze the following similar loan applications for loan ID {loan_id}:
        
        {json.dumps(similar_docs, indent=2)}
        
        Provide insights about:
        1. Common characteristics among similar applications
        2. Factors that might influence loan approval
        3. Patterns in income, education, or other features
        """
        
        analysis = self.generate_response(analysis_prompt, similar_docs)
        
        return {
            "response": analysis,
            "similar_applications": similar_docs,
            "original_loan_id": loan_id
        } 