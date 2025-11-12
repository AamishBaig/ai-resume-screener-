"""
Configuration management module.
"""
import os
import yaml
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file or defaults.
        
        Args:
            config_path: Path to config YAML file
        """
        # Default configuration
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_batch_size = 16
        self.max_files = 50
        self.max_file_size_mb = 10
        self.must_have_keywords = [
            "python", "javascript", "react", "sql", 
            "machine learning", "aws", "docker", "kubernetes"
        ]
        self.keyword_boost_value = 0.05
        self.per_file_timeout = 30
        self.overall_timeout = 300
        
        # Load from file if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self._load_from_dict(config_data)
                    logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        else:
            logger.info("Using default configuration")
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_dict(self, config_data: dict) -> None:
        """Load configuration from dictionary."""
        if not config_data:
            return
        
        self.embedding_model = config_data.get('embedding_model', self.embedding_model)
        self.embedding_batch_size = config_data.get('embedding_batch_size', self.embedding_batch_size)
        self.max_files = config_data.get('max_files', self.max_files)
        self.max_file_size_mb = config_data.get('max_file_size_mb', self.max_file_size_mb)
        self.must_have_keywords = config_data.get('must_have_keywords', self.must_have_keywords)
        self.keyword_boost_value = config_data.get('keyword_boost_value', self.keyword_boost_value)
        self.per_file_timeout = config_data.get('per_file_timeout', self.per_file_timeout)
        self.overall_timeout = config_data.get('overall_timeout', self.overall_timeout)
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        if os.getenv('EMBEDDING_MODEL'):
            self.embedding_model = os.getenv('EMBEDDING_MODEL')
        
        if os.getenv('MAX_FILES'):
            self.max_files = int(os.getenv('MAX_FILES'))
        
        if os.getenv('MAX_FILE_SIZE_MB'):
            self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB'))
