"""
Configuration settings for the Text Summarizer application.
"""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # Summarization settings
    'max_sentences': 50,
    'min_sentence_length': 15,
    'mmr_lambda': 0.7,  # MMR diversity parameter
    
    # File handling
    'max_file_size_mb': 50,
    'supported_formats': ['.txt', '.pdf', '.docx'],
    
    # Gemini API
    'gemini_model': 'gemini-2.5-flash',
    'max_chars_for_gemini': 20000,
    'gemini_timeout': 30,
    
    # UI settings
    'window_size': '1000x800',
    'summary_size_range': (5, 50),
    'default_summary_percent': 20.0,
    
    # Logging
    'log_level': 'INFO',
    'log_file': 'summarizer.log',
    
    # Performance
    'cache_size': 100,
    'enable_caching': True,
}

def get_config() -> Dict[str, Any]:
    """Get configuration with environment variable overrides."""
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    env_mappings = {
        'SUMMARIZER_LOG_LEVEL': 'log_level',
        'SUMMARIZER_MAX_SENTENCES': 'max_sentences',
        'SUMMARIZER_WINDOW_SIZE': 'window_size',
        'GEMINI_API_KEY': 'gemini_api_key',
    }
    
    for env_var, config_key in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion
            if config_key in ['max_sentences', 'max_file_size_mb', 'gemini_timeout']:
                config[config_key] = int(value)
            elif config_key in ['mmr_lambda', 'default_summary_percent']:
                config[config_key] = float(value)
            elif config_key in ['enable_caching']:
                config[config_key] = value.lower() in ('true', '1', 'yes')
            else:
                config[config_key] = value
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values."""
    try:
        assert 0 < config['max_sentences'] <= 1000, "max_sentences must be between 1 and 1000"
        assert 0 < config['min_sentence_length'] <= 100, "min_sentence_length must be between 1 and 100"
        assert 0 <= config['mmr_lambda'] <= 1, "mmr_lambda must be between 0 and 1"
        assert 0 < config['max_file_size_mb'] <= 1000, "max_file_size_mb must be between 1 and 1000"
        assert 0 < config['gemini_timeout'] <= 300, "gemini_timeout must be between 1 and 300"
        return True
    except AssertionError as e:
        print(f"Configuration validation error: {e}")
        return False
