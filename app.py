#!/usr/bin/env python3
"""
Loan Application Chatbot - Main Entry Point
This file serves as the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the simple chatbot
from simple_chatbot import main

if __name__ == "__main__":
    main() 