#!/usr/bin/env python3
"""
Simple Loan Application Chatbot
A lightweight version that works on Streamlit Cloud without ChromaDB
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re

class SimpleLoanChatbot:
    def __init__(self, train_path: str, test_path: str):
        self.train_data = None
        self.test_data = None
        self.load_data(train_path, test_path)
        
    def load_data(self, train_path: str, test_path: str):
        """Load and process the loan data."""
        try:
            # Debug: Print current working directory and file paths
            import os
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for train file: {os.path.abspath(train_path)}")
            print(f"Looking for test file: {os.path.abspath(test_path)}")
            
            # List files in current directory
            print(f"Files in current directory: {os.listdir('.')}")
            
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            print(f"Loaded {len(self.train_data)} training samples and {len(self.test_data)} test samples")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Try alternative paths
            try:
                print("Trying alternative paths...")
                # Try without ./ prefix
                alt_train = train_path.replace('./', '')
                alt_test = test_path.replace('./', '')
                print(f"Trying: {alt_train}, {alt_test}")
                
                self.train_data = pd.read_csv(alt_train)
                self.test_data = pd.read_csv(alt_test)
                print(f"Successfully loaded with alternative paths!")
            except Exception as e2:
                st.error(f"Alternative path also failed: {e2}")
                # Create dummy data for demo
                st.warning("Using demo data since CSV files couldn't be loaded")
                self.train_data = pd.DataFrame({
                    'Loan_ID': ['LP001001', 'LP001002', 'LP001003'],
                    'Gender': ['Male', 'Female', 'Male'],
                    'Married': ['Yes', 'No', 'Yes'],
                    'Dependents': ['0', '0', '1'],
                    'Education': ['Graduate', 'Graduate', 'Not Graduate'],
                    'Self_Employed': ['No', 'No', 'No'],
                    'ApplicantIncome': [5849, 4583, 3000],
                    'CoapplicantIncome': [0, 1508, 0],
                    'LoanAmount': [146, 128, 66],
                    'Loan_Amount_Term': [360, 360, 360],
                    'Credit_History': [1, 1, 1],
                    'Property_Area': ['Urban', 'Rural', 'Urban'],
                    'Loan_Status': ['Y', 'N', 'Y']
                })
                self.test_data = self.train_data.copy()
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if self.train_data is None:
            return {}
            
        stats = {
            'total_applications': len(self.train_data),
            'approved': len(self.train_data[self.train_data['Loan_Status'] == 'Y']),
            'rejected': len(self.train_data[self.train_data['Loan_Status'] == 'N']),
            'approval_rate': round((len(self.train_data[self.train_data['Loan_Status'] == 'Y']) / len(self.train_data)) * 100, 2),
            'avg_income': round(self.train_data['ApplicantIncome'].mean(), 2),
            'avg_loan_amount': round(self.train_data['LoanAmount'].mean(), 2),
            'education_counts': self.train_data['Education'].value_counts().to_dict(),
            'property_area_counts': self.train_data['Property_Area'].value_counts().to_dict(),
            'credit_history_good': len(self.train_data[self.train_data['Credit_History'] == 1]),
            'credit_history_bad': len(self.train_data[self.train_data['Credit_History'] == 0])
        }
        return stats
    
    def search_applications(self, query: str) -> List[Dict[str, Any]]:
        """Simple text-based search through applications."""
        if self.train_data is None:
            return []
            
        query_lower = query.lower()
        results = []
        
        for idx, row in self.train_data.iterrows():
            score = 0
            content = f"{row['Gender']} {row['Married']} {row['Education']} {row['Property_Area']} {row['Loan_Status']}"
            content_lower = content.lower()
            
            # Simple keyword matching
            if any(word in content_lower for word in query_lower.split()):
                score += 1
                
            # Specific field matching
            if 'approved' in query_lower and row['Loan_Status'] == 'Y':
                score += 2
            if 'rejected' in query_lower and row['Loan_Status'] == 'N':
                score += 2
            if 'graduate' in query_lower and row['Education'] == 'Graduate':
                score += 2
            if 'not graduate' in query_lower and row['Education'] == 'Not Graduate':
                score += 2
            if 'urban' in query_lower and row['Property_Area'] == 'Urban':
                score += 2
            if 'rural' in query_lower and row['Property_Area'] == 'Rural':
                score += 2
            if 'semiurban' in query_lower and row['Property_Area'] == 'Semiurban':
                score += 2
                
            if score > 0:
                results.append({
                    'id': row['Loan_ID'],
                    'score': score,
                    'data': row.to_dict(),
                    'content': f"Loan ID: {row['Loan_ID']}, Status: {row['Loan_Status']}, Education: {row['Education']}, Income: ${row['ApplicantIncome']:,.2f}, Property: {row['Property_Area']}"
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]
    
    def answer_question(self, question: str) -> str:
        """Answer questions based on the dataset."""
        question_lower = question.lower()
        stats = self.get_statistics()
        
        # Approval rate questions
        if any(word in question_lower for word in ['approval rate', 'approval', 'approved', 'rejected']):
            return f"Based on the dataset, the loan approval rate is {stats['approval_rate']}% ({stats['approved']} approved out of {stats['total_applications']} total applications)."
        
        # Income questions
        if any(word in question_lower for word in ['income', 'salary', 'earnings']):
            return f"The average applicant income is ${stats['avg_income']:,.2f}. Income ranges from ${self.train_data['ApplicantIncome'].min():,.2f} to ${self.train_data['ApplicantIncome'].max():,.2f}."
        
        # Loan amount questions
        if any(word in question_lower for word in ['loan amount', 'loan size', 'amount']):
            return f"The average loan amount is ${stats['avg_loan_amount']:,.2f}. Loan amounts range from ${self.train_data['LoanAmount'].min():,.2f} to ${self.train_data['LoanAmount'].max():,.2f}."
        
        # Education questions
        if any(word in question_lower for word in ['education', 'graduate', 'degree']):
            edu_text = "Education levels in the dataset:\n"
            for edu, count in stats['education_counts'].items():
                edu_text += f"• {edu}: {count} applications\n"
            return edu_text
        
        # Property area questions
        if any(word in question_lower for word in ['property', 'area', 'urban', 'rural', 'semiurban']):
            area_text = "Property areas in the dataset:\n"
            for area, count in stats['property_area_counts'].items():
                area_text += f"• {area}: {count} applications\n"
            return area_text
        
        # Credit history questions
        if any(word in question_lower for word in ['credit', 'credit history']):
            total_credit = stats['credit_history_good'] + stats['credit_history_bad']
            good_rate = (stats['credit_history_good'] / total_credit * 100) if total_credit > 0 else 0
            return f"Credit history statistics: {good_rate:.1f}% of applicants have good credit history ({stats['credit_history_good']} out of {total_credit} total)."
        
        # Gender questions
        if any(word in question_lower for word in ['gender', 'male', 'female']):
            gender_counts = self.train_data['Gender'].value_counts()
            gender_text = "Gender distribution:\n"
            for gender, count in gender_counts.items():
                gender_text += f"• {gender}: {count} applications\n"
            return gender_text
        
        # Marital status questions
        if any(word in question_lower for word in ['married', 'marital', 'single']):
            marital_counts = self.train_data['Married'].value_counts()
            marital_text = "Marital status distribution:\n"
            for status, count in marital_counts.items():
                marital_text += f"• {status}: {count} applications\n"
            return marital_text
        
        # Dependents questions
        if any(word in question_lower for word in ['dependents', 'children', 'family']):
            dep_counts = self.train_data['Dependents'].value_counts()
            dep_text = "Number of dependents:\n"
            for deps, count in dep_counts.items():
                dep_text += f"• {deps}: {count} applications\n"
            return dep_text
        
        # General statistics
        if any(word in question_lower for word in ['statistics', 'overview', 'summary', 'data']):
            return f"""Dataset Overview:
• Total Applications: {stats['total_applications']}
• Approval Rate: {stats['approval_rate']}%
• Average Income: ${stats['avg_income']:,.2f}
• Average Loan Amount: ${stats['avg_loan_amount']:,.2f}
• Good Credit History: {stats['credit_history_good']} applicants
• Bad Credit History: {stats['credit_history_bad']} applicants"""
        
        # Default response
        return "I can help you with questions about loan approval rates, income statistics, education levels, property areas, credit history, gender distribution, marital status, and dependents. Please ask a specific question!"

def main():
    st.set_page_config(
        page_title="🏦 Loan Application Chatbot",
        page_icon="🏦",
        layout="wide"
    )
    
    st.title("🏦 Loan Application Chatbot")
    st.markdown("Ask questions about loan applications, approval rates, and patterns!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = SimpleLoanChatbot(
                train_path="./Training Dataset.csv",
                test_path="./Test Dataset.csv"
            )
            st.success("✅ Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {e}")
            return
    
    chatbot = st.session_state.chatbot
    
    # Sidebar with statistics
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        stats = chatbot.get_statistics()
        
        if stats:
            st.metric("Total Applications", stats['total_applications'])
            st.metric("Approval Rate", f"{stats['approval_rate']}%")
            st.metric("Average Income", f"${stats['avg_income']:,.0f}")
            st.metric("Average Loan Amount", f"${stats['avg_loan_amount']:,.0f}")
            
            st.subheader("Education Distribution")
            for edu, count in stats['education_counts'].items():
                st.write(f"• {edu}: {count}")
                
            st.subheader("Property Areas")
            for area, count in stats['property_area_counts'].items():
                st.write(f"• {area}: {count}")
    
    # Main chat interface
    st.header("💬 Chat with the Loan Bot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about loan applications..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.answer_question(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sample questions
    st.header("💡 Sample Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is the loan approval rate?"):
            st.session_state.messages.append({"role": "user", "content": "What is the loan approval rate?"})
            response = chatbot.answer_question("What is the loan approval rate?")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        if st.button("What is the average income?"):
            st.session_state.messages.append({"role": "user", "content": "What is the average income?"})
            response = chatbot.answer_question("What is the average income?")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        if st.button("How does education affect approval?"):
            st.session_state.messages.append({"role": "user", "content": "How does education affect approval?"})
            response = chatbot.answer_question("How does education affect approval?")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("What are the property area statistics?"):
            st.session_state.messages.append({"role": "user", "content": "What are the property area statistics?"})
            response = chatbot.answer_question("What are the property area statistics?")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        if st.button("Credit history statistics"):
            st.session_state.messages.append({"role": "user", "content": "Credit history statistics"})
            response = chatbot.answer_question("Credit history statistics")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        if st.button("Show dataset overview"):
            st.session_state.messages.append({"role": "user", "content": "Show dataset overview"})
            response = chatbot.answer_question("Show dataset overview")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 