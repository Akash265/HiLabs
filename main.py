"""
Main processing pipeline for HiLabs Contract Classification System
Fixed version compatible with dashboard and existing modules
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import pandas as pd
import logging
import sys
import os
import traceback
import json
from dataclasses import dataclass, field
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available. Parallel processing disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.error("pdfplumber not available. Please install: pip install pdfplumber")

try:
    from difflib import SequenceMatcher
except ImportError:
    logger.error("difflib not available")


# Data classes for compatibility with dashboard
@dataclass
class ClassificationResult:
    """Result of a single attribute classification"""
    attribute_name: str
    classification: str
    similarity_score: float
    reason: str
    contract_text: str = ""
    template_text: str = ""
    matching_phrases: List[str] = field(default_factory=list)
    detailed_scores: Dict = field(default_factory=dict)
    weighted_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContractClassification:
    """Complete classification results for a contract"""
    contract_id: str
    state: str
    results: List[ClassificationResult] = field(default_factory=list)
    model_type_used: str = "fuzzy"
    
    def add_result(self, result: ClassificationResult):
        """Add a classification result"""
        self.results.append(result)
    
    @property
    def standard_count(self) -> int:
        """Count of standard classifications"""
        return sum(1 for r in self.results if r.classification == "Standard")
    
    @property
    def non_standard_count(self) -> int:
        """Count of non-standard classifications"""
        return sum(1 for r in self.results if r.classification == "Non-Standard")


class ClauseInfo:
    """Information about an extracted clause"""
    def __init__(self, extracted_text: str, location: str = "", confidence: float = 1.0):
        self.extracted_text = extracted_text
        self.location = location
        self.confidence = confidence


class ContractProcessor:
    """Main orchestrator for contract processing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize processor with configuration"""
        self.config = self._load_config(config_path)
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            if not Path(config_path).exists():
                logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Config file {config_path} not found")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # Initialize PDF parser
            self.pdf_parser = PDFParser(self.config.get('pdf_extraction', {}))
            
            # Initialize clause extractor  
            self.clause_extractor = ClauseExtractor(self.config.get('attributes', []))
            
            # Initialize classifier
            model_type = self.config.get('classification', {}).get('model_type', 'fuzzy')
            self.classifier = Classifier(self.config.get('classification', {}))
            
            # Initialize file handler
            self.file_handler = FileHandler(
                self.config['paths']['contracts_dir'],
                self.config['paths']['templates_dir']
            )
            
            # Initialize output manager
            self.output_manager = OutputManager(self.config['paths']['output_dir'])
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_contract(self, contract_path: Path, template_clauses: Dict[str, str], state: str) -> ContractClassification:
        """Process a single contract"""
        contract_id = contract_path.stem
        
        try:
            logger.info(f"Processing contract: {contract_path.name}")
            
            # Extract text
            text = self.pdf_parser.parse_pdf(contract_path)
            if not text:
                logger.error(f"No text extracted from {contract_path.name}")
                return None
            
            # Extract sections
            sections = self.pdf_parser.extract_sections(text)
            
            # Extract clauses
            extracted_clauses = self.clause_extractor.extract_all_clauses(text, sections)
            
            # Classify clauses
            classification = self.classifier.classify_contract(
                contract_id, state, extracted_clauses, template_clauses
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Error processing contract {contract_path.name}: {e}")
            return None
    
    def process_template(self, template_path: Path) -> Dict[str, str]:
        """Process template to extract standard clauses"""
        try:
            logger.info(f"Processing template: {template_path.name}")
            
            text = self.pdf_parser.parse_pdf(template_path)
            if not text:
                logger.error(f"No text extracted from template {template_path.name}")
                return {}
            
            # Extract sections
            sections = self.pdf_parser.extract_sections(text)
            
            # Extract template clauses
            extracted_clauses = self.clause_extractor.extract_all_clauses(text, sections)
            
            # Convert to simple dictionary
            template_clauses = {
                name: clause.extracted_text 
                for name, clause in extracted_clauses.items()
                if clause.extracted_text
            }
            
            logger.info(f"Extracted {len(template_clauses)} template clauses")
            return template_clauses
            
        except Exception as e:
            logger.error(f"Error processing template {template_path.name}: {e}")
            return {}
    
    def process_state(self, state: str) -> List[ContractClassification]:
        """Process all contracts for a state"""
        logger.info(f"Processing state: {state}")
        
        # Get template
        template_path = self.file_handler.get_template_path(state)
        if not template_path:
            logger.error(f"No template found for state {state}")
            return []
        
        # Process template
        template_clauses = self.process_template(template_path)
        if not template_clauses:
            logger.error(f"No template clauses extracted for {state}")
            return []
        
        # Get contracts
        contracts = self.file_handler.get_state_contracts(state)
        if not contracts:
            logger.warning(f"No contracts found for state {state}")
            return []
        
        logger.info(f"Found {len(contracts)} contracts for state {state}")
        
        # Process contracts
        results = []
        for contract_path in contracts:
            result = self.process_contract(contract_path, template_clauses, state)
            if result:
                results.append(result)
        
        logger.info(f"Processed {len(results)} contracts for state {state}")
        return results
    
    def run(self) -> Dict:
        """Run the full processing pipeline"""
        logger.info("Starting contract classification pipeline")
        start_time = time.time()
        
        all_results = []
        
        try:
            # Process each state
            for state in self.config.get('states', ['TN', 'WA']):
                state_results = self.process_state(state)
                all_results.extend(state_results)
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_results)
            
            # Save results
            report_paths = self.output_manager.save_results(all_results, metrics)
            
            processing_time = time.time() - start_time
            
            summary = {
                'total_contracts': len(all_results),
                'processing_time': processing_time,
                'metrics': metrics,
                'report_paths': report_paths,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Processing complete in {processing_time:.2f} seconds")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise
    
    def _calculate_metrics(self, results: List[ContractClassification]) -> Dict:
        """Calculate summary metrics"""
        total_contracts = len(results)
        total_standard = sum(r.standard_count for r in results)
        total_non_standard = sum(r.non_standard_count for r in results)
        total_not_found = sum(
            sum(1 for res in r.results if res.classification == "Not Found")
            for r in results
        )
        
        return {
            'total_contracts': total_contracts,
            'total_standard': total_standard,
            'total_non_standard': total_non_standard,
            'total_not_found': total_not_found,
            'average_confidence': sum(
                res.similarity_score for r in results for res in r.results
            ) / max(sum(len(r.results) for r in results), 1)
        }


class PDFParser:
    """PDF parser using pdfplumber"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def parse_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not available. Please install: pip install pdfplumber")
        
        try:
            full_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            
            return self._clean_text(full_text)
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return ""
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text"""
        sections = {}
        
        # Simple section extraction based on headers
        lines = text.split('\n')
        current_section = "main"
        current_content = []
        
        for line in lines:
            # Check if line looks like a header
            if line.isupper() and len(line) > 5 and len(line) < 100:
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = text.strip()
        
        return text


class ClauseExtractor:
    """Extract clauses from contract text"""
    
    def __init__(self, attributes: List):
        self.attributes = self._process_attributes(attributes)
    
    def _process_attributes(self, attributes: List) -> Dict:
        """Process attribute configuration"""
        attribute_dict = {}
        
        for attr in attributes:
            if isinstance(attr, dict):
                name = attr.get('name', '')
                attribute_dict[name] = attr
            else:
                # Handle simple string attributes
                attribute_dict[str(attr)] = {'name': str(attr)}
        
        return attribute_dict
    
    def extract_all_clauses(self, text: str, sections: Dict[str, str]) -> Dict[str, ClauseInfo]:
        """Extract all clauses from text"""
        extracted = {}
        
        for attr_name, attr_config in self.attributes.items():
            clause_text = self._extract_clause(text, sections, attr_config)
            extracted[attr_name] = ClauseInfo(clause_text)
        
        return extracted
    
    def _extract_clause(self, text: str, sections: Dict[str, str], attr_config: Dict) -> str:
        """Extract a single clause"""
        # Get extraction patterns
        patterns = attr_config.get('extraction_patterns', [])
        section_patterns = attr_config.get('section_patterns', [])
        
        # First try to find in specific sections
        for section_pattern in section_patterns:
            for section_name, section_text in sections.items():
                if re.search(section_pattern, section_name, re.IGNORECASE):
                    # Try extraction patterns in this section
                    for pattern in patterns:
                        match = re.search(pattern, section_text, re.IGNORECASE)
                        if match:
                            # Extract context around match
                            start = max(0, match.start() - 200)
                            end = min(len(section_text), match.end() + 200)
                            return section_text[start:end].strip()
        
        # Fallback to searching entire text
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract context around match
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                return text[start:end].strip()
        
        return ""  # Not found


class Classifier:
    """Contract clause classifier"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model_type', 'fuzzy')
        self.threshold = config.get('thresholds', {}).get('standard', 0.8)
        self.fuzzy_config = config.get('fuzzy', {})
        self.exact_config = config.get('exact', {})
    
    def classify_contract(self, contract_id: str, state: str,
                         extracted_clauses: Dict[str, ClauseInfo],
                         template_clauses: Dict[str, str]) -> ContractClassification:
        """Classify contract clauses"""
        classification = ContractClassification(contract_id, state)
        classification.model_type_used = self.model_type
        
        for attr_name, template_text in template_clauses.items():
            clause_info = extracted_clauses.get(attr_name)
            
            if not clause_info or not clause_info.extracted_text:
                # Clause not found
                result = ClassificationResult(
                    attribute_name=attr_name,
                    classification="Not Found",
                    similarity_score=0.0,
                    reason=f"The '{attr_name}' clause was not found in the contract",
                    contract_text="",
                    template_text=template_text
                )
            else:
                # Calculate scores and classify
                contract_text = clause_info.extracted_text
                scores = self._calculate_scores(contract_text, template_text)
                
                # Determine classification
                weighted_score = scores['weighted_score']
                if weighted_score >= self.threshold:
                    classification_type = "Standard"
                    reason = f"High similarity ({weighted_score:.1%}) with template"
                else:
                    classification_type = "Non-Standard"
                    reason = f"Low similarity ({weighted_score:.1%}) with template"
                
                result = ClassificationResult(
                    attribute_name=attr_name,
                    classification=classification_type,
                    similarity_score=weighted_score,
                    reason=reason,
                    contract_text=contract_text,
                    template_text=template_text,
                    detailed_scores=scores,
                    weighted_score=weighted_score
                )
            
            classification.add_result(result)
        
        return classification
    
    def _calculate_scores(self, contract_text: str, template_text: str) -> Dict:
        """Calculate all scoring metrics"""
        scores = {'model_type': self.model_type}
        
        # Calculate fuzzy score
        fuzzy_score = SequenceMatcher(None, 
                                     contract_text.lower(), 
                                     template_text.lower()).ratio()
        scores['fuzzy_score'] = fuzzy_score
        
        # Calculate exact match score
        contract_words = set(re.findall(r'\b[a-z]+\b', contract_text.lower()))
        template_words = set(re.findall(r'\b[a-z]+\b', template_text.lower()))
        
        if template_words:
            intersection = len(contract_words & template_words)
            union = len(contract_words | template_words)
            exact_score = intersection / union if union > 0 else 0
        else:
            exact_score = 0
        
        scores['exact_score'] = exact_score
        
        # Semantic score (placeholder for fuzzy mode)
        scores['semantic_score'] = 0 if self.model_type == 'fuzzy' else 0
        
        # Calculate weighted score
        if self.model_type == 'fuzzy':
            fuzzy_weight = self.fuzzy_config.get('weight', 0.7)
            exact_weight = self.exact_config.get('weight', 0.3)
            total_weight = fuzzy_weight + exact_weight
            
            if total_weight > 0:
                weighted_score = (fuzzy_score * fuzzy_weight + 
                                exact_score * exact_weight) / total_weight
            else:
                weighted_score = fuzzy_score
        else:
            # For semantic/hybrid, would need actual semantic scoring
            weighted_score = fuzzy_score
        
        scores['weighted_score'] = weighted_score
        
        return scores


class FileHandler:
    """Handle file operations"""
    
    def __init__(self, contracts_dir: str, templates_dir: str):
        self.contracts_dir = Path(contracts_dir)
        self.templates_dir = Path(templates_dir)
        
        # Create directories if they don't exist
        self.contracts_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def get_template_path(self, state: str) -> Optional[Path]:
        """Get template file path for state"""
        # Look for state-specific template
        patterns = [
            f"{state}_*.pdf",
            f"*{state}*.pdf",
            f"{state.lower()}*.pdf",
            f"*{state.lower()}*.pdf"
        ]
        
        for pattern in patterns:
            matches = list(self.templates_dir.glob(pattern))
            if matches:
                return matches[0]
        
        # Fallback to any PDF
        pdfs = list(self.templates_dir.glob("*.pdf"))
        return pdfs[0] if pdfs else None
    
    def get_state_contracts(self, state: str) -> List[Path]:
        """Get contract files for state"""
        contracts = []
        
        # Look in state subdirectory
        state_dir = self.contracts_dir / state
        if state_dir.exists():
            contracts.extend(state_dir.glob("*.pdf"))
        
        # Look for state-prefixed files
        patterns = [
            f"{state}_*.pdf",
            f"*{state}*.pdf",
            f"{state.lower()}*.pdf"
        ]
        
        for pattern in patterns:
            contracts.extend(self.contracts_dir.glob(pattern))
        
        return list(set(contracts))  # Remove duplicates


class OutputManager:
    """Manage output and reporting"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_dir / timestamp
        self.session_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[ContractClassification], metrics: Dict) -> Dict:
        """Save results in various formats"""
        paths = {}
        
        # Save JSON
        json_path = self.session_dir / 'results.json'
        self._save_json(results, metrics, json_path)
        paths['json'] = str(json_path)
        
        # Save CSV
        csv_path = self.session_dir / 'results.csv'
        self._save_csv(results, csv_path)
        paths['csv'] = str(csv_path)
        
        # Save summary
        summary_path = self.session_dir / 'summary.txt'
        self._save_summary(metrics, summary_path)
        paths['summary'] = str(summary_path)
        
        logger.info(f"Results saved to {self.session_dir}")
        return paths
    
    def _save_json(self, results: List[ContractClassification], metrics: Dict, path: Path):
        """Save results as JSON"""
        data = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        for classification in results:
            contract_data = {
                'contract_id': classification.contract_id,
                'state': classification.state,
                'model_type': classification.model_type_used,
                'standard_count': classification.standard_count,
                'non_standard_count': classification.non_standard_count,
                'classifications': []
            }
            
            for result in classification.results:
                contract_data['classifications'].append({
                    'attribute': result.attribute_name,
                    'classification': result.classification,
                    'score': result.similarity_score,
                    'reason': result.reason,
                    'detailed_scores': result.detailed_scores
                })
            
            data['results'].append(contract_data)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_csv(self, results: List[ContractClassification], path: Path):
        """Save results as CSV"""
        rows = []
        
        for classification in results:
            for result in classification.results:
                rows.append({
                    'Contract_ID': classification.contract_id,
                    'State': classification.state,
                    'Attribute': result.attribute_name,
                    'Classification': result.classification,
                    'Score': result.similarity_score,
                    'Reason': result.reason
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    
    def _save_summary(self, metrics: Dict, path: Path):
        """Save summary report"""
        with open(path, 'w') as f:
            f.write("CONTRACT CLASSIFICATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Contracts: {metrics['total_contracts']}\n")
            f.write(f"Standard Clauses: {metrics['total_standard']}\n")
            f.write(f"Non-Standard Clauses: {metrics['total_non_standard']}\n")
            f.write(f"Not Found: {metrics['total_not_found']}\n")
            f.write(f"Average Confidence: {metrics['average_confidence']:.2%}\n")
            f.write(f"\nGenerated: {datetime.now()}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HiLabs Contract Classification System')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--contracts', type=str, help='Override contracts directory')
    parser.add_argument('--templates', type=str, help='Override templates directory')
    parser.add_argument('--output', type=str, help='Override output directory')
    parser.add_argument('--state', type=str, choices=['TN', 'WA'], help='Process specific state')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize processor
        processor = ContractProcessor(args.config)
        
        # Override config with command line arguments
        if args.contracts:
            processor.config['paths']['contracts_dir'] = args.contracts
            processor.file_handler.contracts_dir = Path(args.contracts)
        if args.templates:
            processor.config['paths']['templates_dir'] = args.templates
            processor.file_handler.templates_dir = Path(args.templates)
        if args.output:
            processor.config['paths']['output_dir'] = args.output
            processor.output_manager.output_dir = Path(args.output)
        if args.state:
            processor.config['states'] = [args.state]
        
        # Run processing
        results = processor.run()
        
        # Print summary
        print("\n" + "="*60)
        print("HILABS CONTRACT CLASSIFICATION COMPLETE")
        print("="*60)
        print(f"Total Contracts: {results['total_contracts']}")
        print(f"Processing Time: {results['processing_time']:.2f} seconds")
        
        if results['total_contracts'] > 0:
            metrics = results['metrics']
            print(f"\nClassification Summary:")
            print(f"  Standard: {metrics['total_standard']}")
            print(f"  Non-Standard: {metrics['total_non_standard']}")
            print(f"  Not Found: {metrics['total_not_found']}")
            print(f"  Average Confidence: {metrics['average_confidence']:.2%}")
        
        print(f"\nResults saved to: {results['report_paths']['summary']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())