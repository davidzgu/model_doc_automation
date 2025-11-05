# settings.py

# Configuration settings for the document automation application

import os

# Define constants
DOCUMENTS_DIR = os.getenv('DOCUMENTS_DIR', './documents')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# LangChain settings
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY', 'your_api_key_here')
LANGCHAIN_MODEL = os.getenv('LANGCHAIN_MODEL', 'gpt-3.5-turbo')

# Other settings
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
TIMEOUT = int(os.getenv('TIMEOUT', 30))  # in seconds

# Add any additional configuration settings as needed