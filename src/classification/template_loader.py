"""
Template loader and processor for standard templates
"""

from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd

from src.data_processing.pdf_parser import PDFParser
from src.data_processing.clause_extractor import ClauseExtractor
from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


@dataclass
class StandardTemplate:
    """Container for standard template data"""
    state: str
    file_path: Path
    standard_clauses: Dict[str, str]
    
    def get_clause(self, attribute_name: str) -> Optional[str]:
        """Get specific clause from template"""
        return self.standard_clauses.get(attribute_name)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'state': self.state,
            'file_path': str(self.file_path),
            'clauses': list(self.standard_clauses.keys()),
            'clause_count': len(self.standard_clauses)
        }


class TemplateLoader:
    """Load and process standard templates"""
    
    def __init__(self, pdf_parser: PDFParser, clause_extractor: ClauseExtractor):
        """
        Initialize template loader
        
        Args:
            pdf_parser: PDF parser instance
            clause_extractor: Clause extractor instance
        """
        self.pdf_parser = pdf_parser
        self.clause_extractor = clause_extractor
        self.templates_cache = {}
        
    def load_template(self, state: str, template_path: Path) -> StandardTemplate:
        """
        Load and process a template file
        
        Args:
            state: State code (TN, WA)
            template_path: Path to template PDF
            
        Returns:
            StandardTemplate object
        """
        # Check cache
        cache_key = f"{state}_{template_path.stem}"
        if cache_key in self.templates_cache:
            logger.info(f"Using cached template for {state}")
            return self.templates_cache[cache_key]
            
        logger.info(f"Loading template for state {state} from {template_path}")
        
        # Extract text from PDF
        text = self.pdf_parser.parse_pdf(template_path)
        
        # Extract sections
        sections = self.pdf_parser.extract_sections(text)
        
        # Extract clauses
        extracted_clauses = self.clause_extractor.extract_all_clauses(text, sections)
        
        # Convert to simple dictionary
        standard_clauses = {}
        for attr_name, clause_data in extracted_clauses.items():
            standard_clauses[attr_name] = clause_data.extracted_text
            
        # Create template object
        template = StandardTemplate(
            state=state,
            file_path=template_path,
            standard_clauses=standard_clauses
        )
        
        # Cache for reuse
        self.templates_cache[cache_key] = template
        
        logger.info(f"Loaded {len(standard_clauses)} standard clauses for {state}")
        
        return template
        
    def load_all_templates(self, templates_dir: Path, states: list) -> Dict[str, StandardTemplate]:
        """
        Load all templates for specified states
        
        Args:
            templates_dir: Directory containing template files
            states: List of state codes
            
        Returns:
            Dictionary mapping state to template
        """
        templates = {}
        
        for state in states:
            template_path = templates_dir / f"{state}.pdf"
            
            if not template_path.exists():
                # Try lowercase
                template_path = templates_dir / f"{state.lower()}.pdf"
                
            if template_path.exists():
                templates[state] = self.load_template(state, template_path)
            else:
                logger.warning(f"Template not found for state {state}")
                
        return templates
        
    def validate_template(self, template: StandardTemplate, required_attributes: list) -> tuple[bool, list]:
        """
        Validate that template contains required attributes
        
        Args:
            template: Template to validate
            required_attributes: List of required attribute names
            
        Returns:
            Tuple of (is_valid, missing_attributes)
        """
        missing = []
        
        for attr in required_attributes:
            if attr not in template.standard_clauses:
                missing.append(attr)
                
        is_valid = len(missing) == 0
        
        if not is_valid:
            logger.warning(f"Template for {template.state} missing attributes: {missing}")
            
        return is_valid, missing
        
    def parse_attribute_dictionary(self, dict_path: Path) -> Dict:
        """
        Parse the attribute dictionary Excel file
        
        Args:
            dict_path: Path to Attribute_Dictionary.xlsx
            
        Returns:
            Dictionary of attribute configurations
        """
        try:
            df = pd.read_excel(dict_path)
            
            attributes = {}
            for _, row in df.iterrows():
                attr_name = row['Attribute']
                attributes[attr_name] = {
                    'name': attr_name,
                    'section': row.get('Example Section in Document', ''),
                    'description': row.get('Description', ''),
                    'example': row.get('Example Extracted Language', '')
                }
                
            logger.info(f"Parsed {len(attributes)} attributes from dictionary")
            return attributes
            
        except Exception as e:
            logger.error(f"Error parsing attribute dictionary: {e}")
            return {}
            
    def compare_templates(self, template1: StandardTemplate, template2: StandardTemplate) -> Dict:
        """
        Compare two templates to identify differences
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Comparison results
        """
        comparison = {
            'state1': template1.state,
            'state2': template2.state,
            'common_attributes': [],
            'unique_to_state1': [],
            'unique_to_state2': [],
            'differences': []
        }
        
        attrs1 = set(template1.standard_clauses.keys())
        attrs2 = set(template2.standard_clauses.keys())
        
        # Common attributes
        common = attrs1 & attrs2
        comparison['common_attributes'] = list(common)
        
        # Unique attributes
        comparison['unique_to_state1'] = list(attrs1 - attrs2)
        comparison['unique_to_state2'] = list(attrs2 - attrs1)
        
        # Compare common attributes
        for attr in common:
            text1 = template1.standard_clauses[attr]
            text2 = template2.standard_clauses[attr]
            
            if text1 != text2:
                comparison['differences'].append({
                    'attribute': attr,
                    f'{template1.state}_text': text1[:200],
                    f'{template2.state}_text': text2[:200]
                })
                
        return comparison