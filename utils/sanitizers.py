"""
Input sanitization utilities for security.
"""
import re
import os
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injections.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Get base name only (remove any path components)
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:90] + ext
    
    # Ensure not empty
    if not filename:
        filename = "unnamed_file.pdf"
    
    return filename


def sanitize_text(text: str) -> str:
    """
    Sanitize text content for security.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove script tags and javascript
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text
