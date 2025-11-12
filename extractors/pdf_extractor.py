"""
PDF text extraction module with fallback chain.
Supports PyPDF2, pdfplumber, and OCR via Tesseract.
"""
import logging
from typing import Tuple, Optional
from io import BytesIO
import time

logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not available")


class PDFExtractor:
    """Extract text from PDF files with multiple fallback strategies."""
    
    def __init__(self, config):
        """
        Initialize PDF extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.min_text_length = 50
        self.max_retries = 2
    
    def extract_with_pypdf2(self, file_bytes: BytesIO, filename: str) -> Tuple[str, bool]:
        """
        Extract text using PyPDF2.
        
        Args:
            file_bytes: PDF file as BytesIO
            filename: Name of the file (for logging)
            
        Returns:
            Tuple of (extracted_text, success)
        """
        if not PYPDF2_AVAILABLE:
            return "", False
        
        try:
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"PyPDF2 page {page_num} error in {filename}: {str(e)}")
            
            full_text = "\n".join(text_parts)
            success = len(full_text) >= self.min_text_length
            
            if success:
                logger.info(f"PyPDF2 extracted {len(full_text)} chars from {filename}")
            
            return full_text, success
            
        except Exception as e:
            logger.error(f"PyPDF2 failed for {filename}: {str(e)}")
            return "", False
    
    def extract_with_pdfplumber(self, file_bytes: BytesIO, filename: str) -> Tuple[str, bool]:
        """
        Extract text using pdfplumber.
        
        Args:
            file_bytes: PDF file as BytesIO
            filename: Name of the file (for logging)
            
        Returns:
            Tuple of (extracted_text, success)
        """
        if not PDFPLUMBER_AVAILABLE:
            return "", False
        
        try:
            file_bytes.seek(0)
            text_parts = []
            
            with pdfplumber.open(file_bytes) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"pdfplumber page {page_num} error in {filename}: {str(e)}")
            
            full_text = "\n".join(text_parts)
            success = len(full_text) >= self.min_text_length
            
            if success:
                logger.info(f"pdfplumber extracted {len(full_text)} chars from {filename}")
            
            return full_text, success
            
        except Exception as e:
            logger.error(f"pdfplumber failed for {filename}: {str(e)}")
            return "", False
    
    def extract_with_ocr(self, file_bytes: BytesIO, filename: str) -> Tuple[str, bool]:
        """
        Extract text using OCR (Tesseract).
        
        Args:
            file_bytes: PDF file as BytesIO
            filename: Name of the file (for logging)
            
        Returns:
            Tuple of (extracted_text, success)
        """
        if not OCR_AVAILABLE:
            logger.warning(f"OCR requested but not available for {filename}")
            return "", False
        
        try:
            file_bytes.seek(0)
            
            # Convert PDF to images
            images = convert_from_bytes(
                file_bytes.read(),
                dpi=300,
                fmt='jpeg',
                thread_count=2
            )
            
            text_parts = []
            for page_num, image in enumerate(images):
                try:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"OCR page {page_num} error in {filename}: {str(e)}")
            
            full_text = "\n".join(text_parts)
            success = len(full_text) >= self.min_text_length
            
            if success:
                logger.info(f"OCR extracted {len(full_text)} chars from {filename}")
            
            return full_text, success
            
        except Exception as e:
            logger.error(f"OCR failed for {filename}: {str(e)}")
            return "", False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove potential script-like patterns (security)
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_with_fallback(
        self,
        file_bytes: BytesIO,
        filename: str,
        enable_ocr: bool = False
    ) -> Tuple[str, str]:
        """
        Extract text using fallback chain.
        
        Args:
            file_bytes: PDF file as BytesIO
            filename: Name of the file
            enable_ocr: Whether to enable OCR fallback
            
        Returns:
            Tuple of (extracted_text, method_used)
        """
        start_time = time.time()
        
        # Try PyPDF2 first
        text, success = self.extract_with_pypdf2(file_bytes, filename)
        if success:
            clean_text = self.clean_text(text)
            duration = time.time() - start_time
            logger.info(f"{filename}: PyPDF2 success in {duration:.2f}s")
            return clean_text, "PyPDF2"
        
        # Fallback to pdfplumber
        text, success = self.extract_with_pdfplumber(file_bytes, filename)
        if success:
            clean_text = self.clean_text(text)
            duration = time.time() - start_time
            logger.info(f"{filename}: pdfplumber success in {duration:.2f}s")
            return clean_text, "pdfplumber"
        
        # Fallback to OCR if enabled
        if enable_ocr:
            text, success = self.extract_with_ocr(file_bytes, filename)
            if success:
                clean_text = self.clean_text(text)
                duration = time.time() - start_time
                logger.info(f"{filename}: OCR success in {duration:.2f}s")
                return clean_text, "OCR"
        
        # All methods failed
        duration = time.time() - start_time
        logger.error(f"{filename}: All extraction methods failed in {duration:.2f}s")
        return "", "none"
