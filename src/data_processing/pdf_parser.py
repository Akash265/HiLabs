"""
PDF parsing and text extraction module
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from dataclasses import dataclass

from src.utils.logger import Logger, ProcessingLogger

logger = Logger.setup_logger(__name__)
processing_logger = ProcessingLogger(logger)


@dataclass
class ExtractedPage:
    """Container for extracted page data"""
    page_num: int
    text: str
    sections: Dict[str, str]
    tables: List[List[str]]
    confidence: float = 1.0


class PDFParser:
    """Extract and structure text from PDF files"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize PDF parser
        
        Args:
            config: Parser configuration settings
        """
        self.config = config or {
            'x_tolerance': 3,
            'y_tolerance': 3,
            'keep_blank_chars': True,
            'layout_mode': True
        }
        
    def parse_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        doc_id = file_path.stem
        processing_logger.start_processing(doc_id, file_path.name)
        
        try:
            # Try pdfplumber first (best for text-based PDFs)
            text = self._extract_with_pdfplumber(file_path)
            
            if not text or len(text.strip()) < 100:
                # Fallback to OCR for scanned PDFs
                logger.info(f"Insufficient text from pdfplumber, trying OCR for {file_path.name}")
                text = self._extract_with_ocr(file_path)
                
            processing_logger.end_processing(doc_id, file_path.name, success=True)
            return text
            
        except Exception as e:
            processing_logger.log_error(doc_id, e, "PDF parsing")
            processing_logger.end_processing(doc_id, file_path.name, success=False)
            raise
            
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber"""
        full_text = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract with custom settings
                    page_text = page.extract_text(
                        x_tolerance=self.config['x_tolerance'],
                        y_tolerance=self.config['y_tolerance'],
                        keep_blank_chars=self.config['keep_blank_chars'],
                        layout=self.config['layout_mode']
                    )
                    
                    if page_text:
                        # Add page marker for reference
                        full_text.append(f"\n[PAGE {i+1}]\n")
                        full_text.append(page_text)
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                table_text = self._format_table(table)
                                if table_text:
                                    full_text.append(table_text)
                                    
        except Exception as e:
            logger.error(f"Error with pdfplumber extraction: {e}")
            return ""
            
        return "\n".join(full_text)
        
    def _extract_with_ocr(self, file_path: Path) -> str:
        """Extract text using OCR (for scanned PDFs)"""
        full_text = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                page_text = pytesseract.image_to_string(img, lang='eng')
                
                if page_text:
                    full_text.append(f"\n[PAGE {page_num + 1}]\n")
                    full_text.append(page_text)
                    
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error with OCR extraction: {e}")
            return ""
            
        return "\n".join(full_text)
        
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from document text
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary mapping section headers to content
        """
        sections = {}
        
        # Common section header patterns
        section_patterns = [
            r'^(?P<header>[A-Z][A-Z\s]+)$',  # ALL CAPS HEADERS
            r'^(?P<header>\d+\.[\d\.]*\s+[A-Z][^.]+)$',  # Numbered sections
            r'^(?P<header>Article\s+[IVX]+[\.\:]?\s+.+)$',  # Article headers
            r'^(?P<header>Section\s+\d+[\.\:]?\s+.+)$',  # Section headers
        ]
        
        lines = text.split('\n')
        current_section = "PREAMBLE"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                        
                    # Start new section
                    current_section = match.group('header') if 'header' in match.groupdict() else line
                    current_content = []
                    is_header = True
                    break
                    
            if not is_header:
                current_content.append(line)
                
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        logger.info(f"Extracted {len(sections)} sections from document")
        return sections
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove page markers for processing
        text = re.sub(r'\[PAGE \d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove common headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Confidential.*\n', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')  # Common OCR mistake
        text = re.sub(r'(\d)O(\d)', r'\1\2', text)  # O instead of 0 in numbers
        
        # Remove excessive blank lines
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)
        
    def extract_page_numbers(self, file_path: Path) -> Dict[int, str]:
        """
        Extract text with page number mapping
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to text
        """
        pages = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text(
                        x_tolerance=self.config['x_tolerance'],
                        y_tolerance=self.config['y_tolerance']
                    )
                    if page_text:
                        pages[i + 1] = self.clean_text(page_text)
                        
        except Exception as e:
            logger.error(f"Error extracting pages: {e}")
            
        return pages
        
    def _format_table(self, table: List[List]) -> str:
        """Format extracted table as text"""
        if not table:
            return ""
            
        formatted_rows = []
        for row in table:
            # Filter None values and join cells
            cells = [str(cell) if cell else "" for cell in row]
            formatted_rows.append(" | ".join(cells))
            
        return "\n".join(formatted_rows)
        
    def find_text_location(self, file_path: Path, search_text: str) -> List[Tuple[int, int]]:
        """
        Find location of specific text in PDF
        
        Args:
            file_path: Path to PDF file
            search_text: Text to search for
            
        Returns:
            List of (page_num, char_position) tuples
        """
        locations = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and search_text.lower() in page_text.lower():
                        # Find all occurrences
                        start = 0
                        while True:
                            pos = page_text.lower().find(search_text.lower(), start)
                            if pos == -1:
                                break
                            locations.append((i + 1, pos))
                            start = pos + 1
                            
        except Exception as e:
            logger.error(f"Error finding text location: {e}")
            
        return locations


class PDFValidator:
    """Validate PDF files before processing"""
    
    @staticmethod
    def validate(file_path: Path) -> Tuple[bool, str, Dict]:
        """
        Validate PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}
        
        try:
            # Check file exists
            if not file_path.exists():
                return False, "File does not exist", metadata
                
            # Check file extension
            if file_path.suffix.lower() != '.pdf':
                return False, "File is not a PDF", metadata
                
            # Try to open with pdfplumber
            with pdfplumber.open(file_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                metadata['has_text'] = False
                metadata['has_images'] = False
                
                # Check for text content
                for page in pdf.pages[:5]:  # Check first 5 pages
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        metadata['has_text'] = True
                        break
                        
                # Check file size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                metadata['size_mb'] = size_mb
                
                if size_mb > 50:
                    return False, f"File too large: {size_mb:.1f}MB", metadata
                    
            return True, "", metadata
            
        except Exception as e:
            return False, f"Error validating PDF: {str(e)}", metadata