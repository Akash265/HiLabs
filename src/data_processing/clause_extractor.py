"""
Extract specific contract clauses and attributes
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from difflib import SequenceMatcher

from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


@dataclass
class ClauseData:
    """Container for extracted clause information"""
    attribute_name: str
    section_header: str
    extracted_text: str
    page_number: Optional[int] = None
    char_positions: Optional[Tuple[int, int]] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'attribute_name': self.attribute_name,
            'section_header': self.section_header,
            'extracted_text': self.extracted_text,
            'page_number': self.page_number,
            'char_positions': self.char_positions,
            'confidence': self.confidence
        }


class ClauseExtractor:
    """Extract specific clauses based on attribute patterns"""
    
    def __init__(self, attribute_config: List[Dict]):
        """
        Initialize clause extractor
        
        Args:
            attribute_config: List of attribute configurations from config.yaml
        """
        self.attributes = attribute_config
        self.attribute_patterns = self._compile_patterns()
        
    
    def _compile_patterns(self) -> Dict[str, Dict]:
        """Compile regex patterns for each attribute"""
        patterns = {}
        
        for attr in self.attributes:
            name = attr['name']
            patterns[name] = {
                'section_patterns': [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in attr['section_patterns']
                ],
                'extraction_patterns': [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in attr['extraction_patterns']
                ],
                'typical_values': attr.get('typical_values', {})
            }
            
        return patterns
        # In clause_extractor.py, add this method to the ClauseExtractor class:

    def clean_text_for_comparison(self, text: str) -> str:
        """Remove non-substantive elements before classification comparison"""
        if not text:
            return ""
        
        # Remove copyright lines entirely
        text = re.sub(r'©\s*\d{4}.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
        
        # Remove dates in various formats
        text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
        text = re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{4}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)', '', text, flags=re.IGNORECASE)
        
        # Remove version numbers (V5.1, v2.0, etc.)
        text = re.sub(r'[Vv]\d+\.\d+', '', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\[PAGE\s*\d+\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)
        
        # Remove document IDs that look like [###########]
        text = re.sub(r'\[#{5,}\]', '', text)
        
        # Remove common footer/header patterns
        text = re.sub(r'Washington Enterprise Provider Agreement.*?(?:\n|$)', '', text)
        text = re.sub(r'© \d{4} July.*?(?:\n|$)', '', text)
        
        # Normalize remaining whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    def extract_all_clauses(self, text: str, sections: Dict[str, str] = None) -> Dict[str, ClauseData]:
        """Extract all configured clauses from document"""
        extracted_clauses = {}
        
        for attr_name in self.attribute_patterns:
            clause_data = self.extract_clause(text, attr_name, sections)
            if clause_data:
                # Clean the extracted text for comparison
                clause_data.extracted_text = self.clean_text_for_comparison(clause_data.extracted_text)
                extracted_clauses[attr_name] = clause_data
                logger.info(f"Extracted {attr_name}: {clause_data.extracted_text[:100]}...")
            else:
                logger.warning(f"Could not extract {attr_name}")
                
        return extracted_clauses
        
    def extract_clause(
        self, 
        text: str, 
        attribute: str, 
        sections: Dict[str, str] = None
    ) -> Optional[ClauseData]:
        """
        Extract specific clause for an attribute
        
        Args:
            text: Document text
            attribute: Attribute name to extract
            sections: Optional pre-extracted sections
            
        Returns:
            Extracted clause data or None
        """
        if attribute not in self.attribute_patterns:
            logger.error(f"Unknown attribute: {attribute}")
            return None
            
        patterns = self.attribute_patterns[attribute]
        
        # Special handling for fee schedule attributes
        if "Fee Schedule" in attribute:
            return self._extract_fee_schedule(text, attribute, patterns, sections)
            
        # Try to find section first
        section_text, section_header = self._find_section(text, patterns['section_patterns'], sections)
        
        if not section_text:
            # Search entire document as fallback
            section_text = text
            section_header = "Full Document"
            
        # Extract specific clause using patterns
        for pattern in patterns['extraction_patterns']:
            match = pattern.search(section_text)
            if match:
                # Extract surrounding context
                extracted = self._extract_context(section_text, match)
                
                return ClauseData(
                    attribute_name=attribute,
                    section_header=section_header,
                    extracted_text=extracted,
                    confidence=0.9  # High confidence for pattern match
                )
                
        # Fallback: fuzzy search for typical values
        typical = patterns['typical_values']
        if typical:
            found_text = self._fuzzy_search_typical_values(section_text, typical)
            if found_text:
                return ClauseData(
                    attribute_name=attribute,
                    section_header=section_header,
                    extracted_text=found_text,
                    confidence=0.7  # Lower confidence for fuzzy match
                )
                
        return None
        
    def _find_section(
        self, 
        text: str, 
        section_patterns: List[re.Pattern], 
        sections: Dict[str, str] = None
    ) -> Tuple[str, str]:
        """
        Find section containing the attribute
        
        Returns:
            Tuple of (section_text, section_header)
        """
        # If sections already extracted, search them first
        if sections:
            for header, content in sections.items():
                for pattern in section_patterns:
                    if pattern.search(header):
                        return content, header
                        
        # Otherwise, search in full text
        for pattern in section_patterns:
            match = pattern.search(text)
            if match:
                # Extract section starting from match
                start = match.start()
                # Find next section header or end of document
                next_section = re.search(
                    r'\n(?:[A-Z][A-Z\s]+|Article\s+[IVX]+|Section\s+\d+)',
                    text[start + len(match.group()):]
                )
                
                if next_section:
                    end = start + len(match.group()) + next_section.start()
                else:
                    end = len(text)
                    
                section_text = text[start:end]
                return section_text, match.group()
                
        return "", ""
        
    def _extract_fee_schedule(
        self, 
        text: str, 
        attribute: str, 
        patterns: Dict, 
        sections: Dict[str, str] = None
    ) -> Optional[ClauseData]:
        """
        Special extraction logic for fee schedule attributes
        
        Fee schedules often appear in "Specific Reimbursement Terms" sections
        """
        # Look for Specific Reimbursement Terms section
        reimbursement_section = ""
        section_header = ""
        
        if sections:
            for header, content in sections.items():
                if "reimbursement" in header.lower() or "compensation" in header.lower():
                    reimbursement_section = content
                    section_header = header
                    break
                    
        if not reimbursement_section:
            # Search in full text
            reimbursement_pattern = re.compile(
                r'(Specific Reimbursement Terms|Plan Compensation Schedule|'
                r'Reimbursement Methodology)[^:]*:?\s*(.*?)(?=\n[A-Z]|\Z)',
                re.IGNORECASE | re.DOTALL
            )
            match = reimbursement_pattern.search(text)
            if match:
                reimbursement_section = match.group(2)
                section_header = match.group(1)
                
        if reimbursement_section:
            # Look for specific fee schedule language
            for pattern in patterns['extraction_patterns']:
                match = pattern.search(reimbursement_section)
                if match:
                    extracted = self._extract_context(reimbursement_section, match)
                    
                    # Try to extract percentage value
                    percent_match = re.search(r'(\d+)\s*%', extracted)
                    if percent_match:
                        extracted = f"{percent_match.group(1)}% of Fee Schedule"
                        
                    return ClauseData(
                        attribute_name=attribute,
                        section_header=section_header,
                        extracted_text=extracted,
                        confidence=0.85
                    )
                    
        return None
        
    def _extract_context(self, text: str, match: re.Match, context_chars: int = 200) -> str:
        """
        Extract matched text with surrounding context
        
        Args:
            text: Source text
            match: Regex match object
            context_chars: Number of context characters to include
            
        Returns:
            Extracted text with context
        """
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        
        # Find sentence boundaries
        before_text = text[start:match.start()]
        after_text = text[match.end():end]
        
        # Find start of sentence before match
        sentence_start = before_text.rfind('.')
        if sentence_start != -1:
            before_text = before_text[sentence_start + 1:]
            
        # Find end of sentence after match
        sentence_end = after_text.find('.')
        if sentence_end != -1:
            after_text = after_text[:sentence_end + 1]
            
        return (before_text + match.group() + after_text).strip()
        
    def _fuzzy_search_typical_values(self, text: str, typical_values: Dict) -> Optional[str]:
        """
        Search for typical values using fuzzy matching
        
        Args:
            text: Text to search
            typical_values: Dictionary of typical values by state
            
        Returns:
            Found text or None
        """
        for state, typical in typical_values.items():
            # Create variations of typical value
            variations = [
                typical,
                typical.replace(" days", ""),
                typical.replace("% of", "% of the"),
                typical.replace("Fee Schedule", "fee schedule")
            ]
            
            for variation in variations:
                # Use SequenceMatcher for fuzzy matching
                words = variation.split()
                pattern = r'\b'.join(re.escape(word) for word in words)
                fuzzy_pattern = re.compile(pattern, re.IGNORECASE)
                
                match = fuzzy_pattern.search(text)
                if match:
                    return self._extract_context(text, match, 100)
                    
        return None


class AttributeValidator:
    """Validate extracted attributes"""
    
    @staticmethod
    def validate_timely_filing(text: str, attribute: str) -> Tuple[bool, str, Dict]:
        """
        Validate timely filing clause
        
        Returns:
            Tuple of (is_valid, error_msg, extracted_values)
        """
        extracted_values = {}
        
        # Extract number of days
        days_match = re.search(r'(\d+)\s*(?:days?|calendar days?)', text, re.IGNORECASE)
        if days_match:
            days = int(days_match.group(1))
            extracted_values['days'] = days
            
            # Validate range
            if attribute == "Medicaid Timely Filing":
                if days < 30 or days > 365:
                    return False, f"Unusual filing period: {days} days", extracted_values
            elif attribute == "Medicare Timely Filing":
                if days < 30 or days > 365:
                    return False, f"Unusual filing period: {days} days", extracted_values
                    
            return True, "", extracted_values
            
        return False, "Could not extract filing period", extracted_values
        
    @staticmethod
    def validate_fee_schedule(text: str, attribute: str) -> Tuple[bool, str, Dict]:
        """
        Validate fee schedule clause
        
        Returns:
            Tuple of (is_valid, error_msg, extracted_values)
        """
        extracted_values = {}
        
        # Extract percentage
        percent_match = re.search(r'(\d+)\s*%', text)
        if percent_match:
            percentage = int(percent_match.group(1))
            extracted_values['percentage'] = percentage
            
            # Validate range
            if percentage < 50 or percentage > 150:
                return False, f"Unusual percentage: {percentage}%", extracted_values
                
            # Check for carve-outs or exceptions
            if re.search(r'except|excluding|other than|carve[\s-]?out', text, re.IGNORECASE):
                extracted_values['has_exceptions'] = True
                return True, "Contains exceptions or carve-outs", extracted_values
                
            return True, "", extracted_values
            
        # Check for alternative fee schedule references
        if re.search(r'lesser of|greater of|charges', text, re.IGNORECASE):
            extracted_values['alternative_methodology'] = True
            return True, "Uses alternative fee methodology", extracted_values
            
        return False, "Could not extract fee schedule percentage", extracted_values