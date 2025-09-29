"""
Main processing pipeline for HiLabs Contract Classification System
Corrected version with proper error handling and simplified structure
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

# Core components - will be defined inline to avoid import issues
class ContractProcessor:
    """Main orchestrator for contract processing with inline components"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize processor with configuration"""
        self.config = self._load_config(config_path)
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            if not Path(config_path).exists():
                # Create default config if not exists
                default_config = self._get_default_config()
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default configuration at {config_path}")
                return default_config
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'paths': {
                'contracts_dir': 'data/contracts',
                'templates_dir': 'data/templates', 
                'output_dir': 'output'
            },
            'states': ['TN', 'WA'],
            'attributes': [
                'Medicaid Timely Filing',
                'Medicare Timely Filing',
                'No Steerage/SOC',
                'Medicaid Fee Schedule',
                'Medicare Fee Schedule'
            ],
            'classification': {
                'model_type': 'fuzzy',
                'thresholds': {
                    'standard': 0.8,
                    'non_standard': 0.7
                }
            },
            'performance': {
                'parallel_workers': 1
            }
        }
        
    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # Initialize simple PDF parser
            self.pdf_parser = SimplePDFParser()
            
            # Initialize clause extractor
            self.clause_extractor = SimpleClauseExtractor(self.config['attributes'])
            
            # Initialize classifier
            self.classifier = SimpleClassifier(self.config['classification'])
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_contract(self, contract_path: Path, template_clauses: Dict[str, str], state: str) -> Optional[Dict]:
        """Process a single contract"""
        contract_id = contract_path.stem
        
        try:
            logger.info(f"Processing contract: {contract_path.name}")
            
            # Extract text
            text = self.pdf_parser.parse_pdf(contract_path)
            if not text:
                logger.error(f"No text extracted from {contract_path.name}")
                return None
            
            # Extract clauses
            extracted_clauses = self.clause_extractor.extract_clauses(text)
            
            # Classify clauses
            classification_results = self.classifier.classify_contract(
                contract_id, state, extracted_clauses, template_clauses
            )
            
            return {
                'contract_id': contract_id,
                'contract_path': str(contract_path),
                'state': state,
                'results': classification_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing contract {contract_path.name}: {e}")
            return {
                'contract_id': contract_id,
                'contract_path': str(contract_path),
                'state': state,
                'results': [],
                'status': 'error',
                'error': str(e)
            }
    
    def process_template(self, template_path: Path) -> Dict[str, str]:
        """Process template to extract standard clauses"""
        try:
            logger.info(f"Processing template: {template_path.name}")
            
            text = self.pdf_parser.parse_pdf(template_path)
            if not text:
                logger.error(f"No text extracted from template {template_path.name}")
                return {}
            
            # Extract template clauses
            template_clauses = self.clause_extractor.extract_clauses(text)
            
            logger.info(f"Extracted {len(template_clauses)} template clauses")
            return template_clauses
            
        except Exception as e:
            logger.error(f"Error processing template {template_path.name}: {e}")
            return {}
    
    def run(self) -> Dict:
        """Run the full processing pipeline"""
        logger.info("Starting contract classification pipeline")
        start_time = time.time()
        
        all_results = []
        
        try:
            # Process each state
            for state in self.config.get('states', ['TN']):
                state_results = self.process_state(state)
                all_results.extend(state_results)
            
            # Generate summary
            summary = self._generate_summary(all_results, time.time() - start_time)
            
            # Save results
            self._save_results(all_results, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise
    
    def process_state(self, state: str) -> List[Dict]:
        """Process all contracts for a state"""
        logger.info(f"Processing state: {state}")
        
        # Find template
        template_path = self._find_template(state)
        if not template_path:
            logger.error(f"No template found for state {state}")
            return []
        
        # Process template
        template_clauses = self.process_template(template_path)
        if not template_clauses:
            logger.error(f"No template clauses extracted for {state}")
            return []
        
        # Find contracts
        contracts = self._find_contracts(state)
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
        
        return results
    
    def _find_template(self, state: str) -> Optional[Path]:
        """Find template file for state"""
        templates_dir = Path(self.config['paths']['templates_dir'])
        
        # Look for state-specific template
        for pattern in [f"{state}_*.pdf", f"*{state}*.pdf", f"{state.lower()}*.pdf"]:
            matches = list(templates_dir.glob(pattern))
            if matches:
                return matches[0]
        
        # Fallback to any PDF in templates directory
        templates = list(templates_dir.glob("*.pdf"))
        return templates[0] if templates else None
    
    def _find_contracts(self, state: str) -> List[Path]:
        """Find contract files for state"""
        contracts_dir = Path(self.config['paths']['contracts_dir'])
        
        contracts = []
        
        # Look in state subdirectory
        state_dir = contracts_dir / state
        if state_dir.exists():
            contracts.extend(state_dir.glob("*.pdf"))
        
        # Look for state-prefixed files in main directory
        contracts.extend(contracts_dir.glob(f"{state}_*.pdf"))
        contracts.extend(contracts_dir.glob(f"*{state}*.pdf"))
        
        return list(set(contracts))  # Remove duplicates
    
    def _generate_summary(self, results: List[Dict], processing_time: float) -> Dict:
        """Generate processing summary"""
        total_contracts = len(results)
        successful_contracts = sum(1 for r in results if r['status'] == 'success')
        
        # Calculate classification metrics
        total_standard = 0
        total_non_standard = 0
        total_not_found = 0
        
        for result in results:
            if result['status'] == 'success':
                for classification in result['results']:
                    if classification['classification'] == 'Standard':
                        total_standard += 1
                    elif classification['classification'] == 'Non-Standard':
                        total_non_standard += 1
                    else:
                        total_not_found += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'total_contracts': total_contracts,
            'successful_contracts': successful_contracts,
            'failed_contracts': total_contracts - successful_contracts,
            'total_standard': total_standard,
            'total_non_standard': total_non_standard,
            'total_not_found': total_not_found,
            'results': results
        }
    
    def _save_results(self, results: List[Dict], summary: Dict):
        """Save results to files"""
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        import json
        with open(output_dir / f'detailed_results_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV
        csv_data = []
        for result in results:
            if result['status'] == 'success':
                for classification in result['results']:
                    csv_data.append({
                        'Contract_ID': result['contract_id'],
                        'State': result['state'],
                        'Attribute': classification['attribute'],
                        'Classification': classification['classification'],
                        'Similarity_Score': classification.get('similarity_score', 0),
                        'Reasoning': classification.get('reasoning', ''),
                        'Contract_Text': classification.get('contract_text', '')[:500],  # First 500 chars
                        'Template_Text': classification.get('template_text', '')[:500]
                    })
            else:
                csv_data.append({
                    'Contract_ID': result['contract_id'],
                    'State': result['state'],
                    'Attribute': 'ERROR',
                    'Classification': 'Failed',
                    'Similarity_Score': 0,
                    'Reasoning': result.get('error', 'Unknown error'),
                    'Contract_Text': '',
                    'Template_Text': ''
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_dir / f'classification_results_{timestamp}.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")


class SimplePDFParser:
    """Simple PDF parser using pdfplumber"""
    
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
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = text.strip()
        
        return text


class SimpleClauseExtractor:
    """Simple clause extractor using keyword matching"""
    
    def __init__(self, attributes: List[str]):
        self.attributes = attributes
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, List[str]]:
        """Build keyword patterns for each attribute"""
        return {
            'Medicaid Timely Filing': [
                'medicaid', 'claims', '120', 'days', 'submit', 'timely', 'filing'
            ],
            'Medicare Timely Filing': [
                'medicare', 'claims', '90', 'days', 'submit', 'timely', 'filing'
            ],
            'No Steerage/SOC': [
                'networks', 'provider', 'steerage', 'steering', 'attachment', 'participating'
            ],
            'Medicaid Fee Schedule': [
                'medicaid', 'fee', 'schedule', '100%', 'reimbursement', 'compensation'
            ],
            'Medicare Fee Schedule': [
                'medicare', 'fee', 'schedule', 'advantage', 'rate', 'reimbursement'
            ]
        }
    
    def extract_clauses(self, text: str) -> Dict[str, str]:
        """Extract clauses for each attribute"""
        results = {}
        text_lower = text.lower()
        
        for attribute in self.attributes:
            keywords = self.patterns.get(attribute, [])
            
            # Find keyword positions
            keyword_positions = []
            for keyword in keywords:
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    keyword_positions.append(pos)
                    start = pos + 1
            
            if keyword_positions:
                # Find the position with most keywords nearby
                best_pos = self._find_best_position(keyword_positions, text)
                
                # Extract context around best position
                start = max(0, best_pos - 200)
                end = min(len(text), best_pos + 200)
                context = text[start:end].strip()
                
                results[attribute] = context
            else:
                results[attribute] = "[Not Found]"
        
        return results
    
    def _find_best_position(self, positions: List[int], text: str) -> int:
        """Find position with highest keyword density"""
        if not positions:
            return 0
        
        # Simple approach: return middle position
        positions.sort()
        return positions[len(positions) // 2]


class SimpleClassifier:
    """Simple classifier using text similarity"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('thresholds', {}).get('standard', 0.8)
    
    def classify_contract(self, contract_id: str, state: str, 
                         extracted_clauses: Dict[str, str], 
                         template_clauses: Dict[str, str]) -> List[Dict]:
        """Classify extracted clauses against template"""
        results = []
        
        for attribute, contract_text in extracted_clauses.items():
            template_text = template_clauses.get(attribute, "")
            
            if contract_text == "[Not Found]":
                classification = "Not Found"
                similarity = 0.0
                reasoning = f"The '{attribute}' clause was not found in the contract"
            elif not template_text:
                classification = "Template Error"
                similarity = 0.0
                reasoning = f"No template available for '{attribute}'"
            else:
                # Calculate simple similarity
                similarity = self._calculate_similarity(contract_text, template_text)
                
                if similarity >= self.threshold:
                    classification = "Standard"
                    reasoning = f"High similarity ({similarity:.1%}) with template"
                else:
                    classification = "Non-Standard"
                    reasoning = f"Low similarity ({similarity:.1%}) with template"
            
            results.append({
                'attribute': attribute,
                'classification': classification,
                'similarity_score': similarity,
                'reasoning': reasoning,
                'contract_text': contract_text,
                'template_text': template_text
            })
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


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
        if args.templates:
            processor.config['paths']['templates_dir'] = args.templates
        if args.output:
            processor.config['paths']['output_dir'] = args.output
        if args.state:
            processor.config['states'] = [args.state]
        
        # Run processing
        results = processor.run()
        
        # Print summary
        print("\n" + "="*60)
        print("HILABS CONTRACT CLASSIFICATION COMPLETE")
        print("="*60)
        print(f"Total Contracts: {results['total_contracts']}")
        print(f"Successful: {results['successful_contracts']}")
        print(f"Failed: {results['failed_contracts']}")
        print(f"Processing Time: {results['processing_time']:.2f} seconds")
        
        if results['successful_contracts'] > 0:
            print(f"\nClassification Summary:")
            print(f"  Standard: {results['total_standard']}")
            print(f"  Non-Standard: {results['total_non_standard']}")
            print(f"  Not Found: {results['total_not_found']}")
        
        print(f"\nResults saved to: {processor.config['paths']['output_dir']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())