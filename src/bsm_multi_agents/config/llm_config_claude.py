# -*- coding: utf-8 -*-
import os
from langchain_anthropic import ChatAnthropic

# ========== CONFIGURATION ==========
# Set your preferences here (can be overridden by environment variables)

# LLM Provider: "claude" (Anthropic)
DEFAULT_PROVIDER = "claude"

# Claude/Anthropic settings
DEFAULT_CLAUDE_MODEL = "claude-3-7-sonnet-latest"

# WARNING: DO NOT commit your real API key to Git!
# Instead, set ANTHROPIC_API_KEY environment variable or create a .env file
DEFAULT_ANTHROPIC_API_KEY = "Nonw"
# ===================================


def get_llm():
    """
    Get LLM based on configuration (Claude/Anthropic).

    Usage:
    1. Set ANTHROPIC_API_KEY environment variable: export ANTHROPIC_API_KEY=sk-ant-...
    2. Or set DEFAULT_ANTHROPIC_API_KEY in llm_config_claude.py (NOT recommended for Git)
    """
    provider = DEFAULT_PROVIDER#os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)

    if provider == "claude":
        api_key = DEFAULT_ANTHROPIC_API_KEY#os.getenv("ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_API_KEY)
        print("claude")
        if not api_key or api_key == "None":
            raise ValueError(
                "ANTHROPIC_API_KEY not provided"
                )

        return ChatAnthropic(
            model=DEFAULT_CLAUDE_MODEL,
            temperature=0,
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. This config only supports 'claude'."
        )
