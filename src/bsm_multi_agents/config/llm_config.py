# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# ========== CONFIGURATION ==========
# Set your preferences here (can be overridden by environment variables)

# LLM Provider: "ollama" or "openai" or "google"
DEFAULT_PROVIDER = "ollama"

# Ollama settings
DEFAULT_OLLAMA_MODEL = "qwen3:32b"

# OpenAI settings
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Google settings
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"

# WARNING: DO NOT commit your real API key to Git!
# Instead, set OPENAI_API_KEY environment variable or create a .env file
DEFAULT_OPENAI_API_KEY = None  # Set to your key for local testing ONLY
DEFAULT_GOOGLE_API_KEY = None
# ===================================


def get_llm(provider: str = DEFAULT_PROVIDER):
    """
    Get LLM based on configuration.

    Usage:
    1. Use Ollama (default): Just run the script
    2. Use OpenAI: Set environment variable LLM_PROVIDER=openai
    3. Use Google: Set environment variable LLM_PROVIDER=google
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either:\n"
                "1. Set environment variable: export OPENAI_API_KEY=sk-...\n"
                "2. Or set DEFAULT_OPENAI_API_KEY in llm_config.py (NOT recommended for Git)"
            )

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            temperature=0,
            api_key=api_key,
        )
    elif provider == "google":
        from langchain_google_genai import HarmCategory, HarmBlockThreshold
        api_key = os.getenv("GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Either:\n"
                "1. Set environment variable: export GOOGLE_API_KEY=...\n"
                "2. Or set DEFAULT_GOOGLE_API_KEY in llm_config.py (NOT recommended for Git)"
            )
        
        # Disable safety filters to prevent "Empty Response" errors during quant analysis
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL),
            temperature=0,
            google_api_key=api_key,
            safety_settings=safety_settings
        )
    else:
        # Default: Ollama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            temperature=0,
        )


class TextOnlyChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config=config, **kwargs)
        
        if isinstance(response.content, list):
            text_content = "".join(
                [part if isinstance(part, str) else part.get("text", "") 
                 for part in response.content]
            )
            response.content = text_content
            
        return response