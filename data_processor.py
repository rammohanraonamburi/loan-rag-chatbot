import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

class LoanDataProcessor:
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize the loan data processor.
        
        Args:
            train_path: Path to training dataset
            test_path: Path to test dataset
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.combined_data = None
        
    def load_data(self) -> None:
        """Load training and test datasets."""
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            print(f"Loaded {len(self.train_data)} training samples and {len(self.test_data)} test samples")
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def clean_data(self) -> None:
        """Clean and preprocess the data."""
        self.train_data = self.train_data.fillna('Unknown')
        self.test_data = self.test_data.fillna('Unknown')
        
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Credit_History', 'Property_Area']
        
        for col in categorical_cols:
            if col in self.train_data.columns:
                self.train_data[col] = self.train_data[col].astype(str)
            if col in self.test_data.columns:
                self.test_data[col] = self.test_data[col].astype(str)
                
        print("Data cleaning completed")
        
    def create_documents(self) -> List[Dict[str, Any]]:
        """Create documents for RAG system from the loan data."""
        documents = []
        
        for idx, row in self.train_data.iterrows():
            doc = {
                'id': f"train_{idx}",
                'content': self._create_loan_description(row),
                'metadata': {
                    'type': 'training',
                    'loan_id': row['Loan_ID'],
                    'loan_status': row.get('Loan_Status', 'Unknown'),
                    'gender': row['Gender'],
                    'education': row['Education'],
                    'property_area': row['Property_Area'],
                    'applicant_income': row['ApplicantIncome'],
                    'loan_amount': row['LoanAmount'],
                    'credit_history': row['Credit_History']
                }
            }
            documents.append(doc)
            
        for idx, row in self.test_data.iterrows():
            doc = {
                'id': f"test_{idx}",
                'content': self._create_loan_description(row),
                'metadata': {
                    'type': 'test',
                    'loan_id': row['Loan_ID'],
                    'gender': row['Gender'],
                    'education': row['Education'],
                    'property_area': row['Property_Area'],
                    'applicant_income': row['ApplicantIncome'],
                    'loan_amount': row['LoanAmount'],
                    'credit_history': row['Credit_History']
                }
            }
            documents.append(doc)
            
        return documents
    
    def _create_loan_description(self, row: pd.Series) -> str:
        """Create a natural language description of a loan application."""
        def safe_format(value, format_str="{}"):
            try:
                if pd.isna(value) or value == '' or value == 'Unknown':
                    return 'Unknown'
                if isinstance(value, (int, float)):
                    return format_str.format(value)
                return str(value)
            except:
                return str(value)
        
        description = f"""
        Loan Application ID: {row['Loan_ID']}
        
        Applicant Profile:
        - Gender: {row['Gender']}
        - Marital Status: {row['Married']}
        - Number of Dependents: {row['Dependents']}
        - Education Level: {row['Education']}
        - Self Employed: {row['Self_Employed']}
        
        Financial Information:
        - Applicant Income: ${safe_format(row['ApplicantIncome'], '{:,.2f}')}
        - Co-applicant Income: ${safe_format(row['CoapplicantIncome'], '{:,.2f}')}
        - Loan Amount: ${safe_format(row['LoanAmount'], '{:,.2f}')}
        - Loan Term: {safe_format(row['Loan_Amount_Term'])} months
        - Credit History: {row['Credit_History']}
        - Property Area: {row['Property_Area']}
        """
        
        if 'Loan_Status' in row:
            description += f"- Loan Status: {row['Loan_Status']}"
            
        return description.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics for the chatbot."""
        stats = {
            'total_applications': len(self.train_data) + len(self.test_data),
            'training_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'approval_rate': None,
            'income_stats': {},
            'education_distribution': {},
            'property_area_distribution': {}
        }
        
        if 'Loan_Status' in self.train_data.columns:
            approval_rate = (self.train_data['Loan_Status'] == 'Y').mean()
            stats['approval_rate'] = f"{approval_rate:.2%}"
            
        stats['income_stats'] = {
            'mean': self.train_data['ApplicantIncome'].mean(),
            'median': self.train_data['ApplicantIncome'].median(),
            'min': self.train_data['ApplicantIncome'].min(),
            'max': self.train_data['ApplicantIncome'].max()
        }
        
        stats['education_distribution'] = self.train_data['Education'].value_counts().to_dict()
        
        stats['property_area_distribution'] = self.train_data['Property_Area'].value_counts().to_dict()
        
        return stats
    
    def create_qa_pairs(self) -> List[Dict[str, str]]:
        """Create sample Q&A pairs for the RAG system."""
        qa_pairs = [
            {
                "question": "What is the overall loan approval rate?",
                "answer": "The loan approval rate can be calculated from the training data."
            },
            {
                "question": "What are the income requirements for loan approval?",
                "answer": "Income requirements vary based on multiple factors including education, property area, and credit history."
            },
            {
                "question": "How does education level affect loan approval?",
                "answer": "Education level is one of the key factors considered in loan approval decisions."
            },
            {
                "question": "What is the difference between urban, semiurban, and rural property areas?",
                "answer": "Property area classification affects loan approval rates and terms."
            },
            {
                "question": "How does credit history impact loan decisions?",
                "answer": "Credit history is a crucial factor in determining loan approval and terms."
            }
        ]
        return qa_pairs 