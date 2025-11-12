"""
File handling utilities.
"""
from typing import Tuple, Any


def validate_file(file: Any, config: Any) -> Tuple[bool, str]:
    """
    Validate uploaded file.
    
    Args:
        file: Uploaded file object
        config: Configuration object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file type
    if not file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are allowed"
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset
    
    max_size = config.max_file_size_mb * 1024 * 1024  # Convert to bytes
    
    if file_size > max_size:
        return False, f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: {config.max_file_size_mb}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, ""
