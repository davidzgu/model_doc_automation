# -*- coding: utf-8 -*-
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ========== CONFIGURATION ==========
# Set your preferences here (can be overridden by environment variables)

# LLM Provider: "ollama" or "openai"
DEFAULT_PROVIDER = "ollama"

# Ollama settings
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# OpenAI settings
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# WARNING: DO NOT commit your real API key to Git!
# Instead, set OPENAI_API_KEY environment variable or create a .env file
DEFAULT_OPENAI_API_KEY = None  # Set to your key for local testing ONLY

# ===================================


def get_llm():
    """
    Get LLM based on configuration.

    Usage:
    1. Use Ollama (default): Just run the script
    2. Use OpenAI: Set environment variable LLM_PROVIDER=openai
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
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
    else:
        # Default: Ollama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            temperature=0,
        )