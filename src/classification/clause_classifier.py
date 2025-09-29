"""
Main classification engine for contract clauses
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.classification.text_comparator import TextComparator, SimilarityScores
from src.data_processing.clause_extractor import ClauseData
from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


@dataclass
class ClassificationResult:
    """Container for classification results"""
    attribute_name: str
    classification: str  # "Standard" or "Non-Standard"
    similarity_score: float
    reason: str
    detailed_scores: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    contract_text: str = ""
    template_text: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'attribute_name': self.attribute_name,
            'classification': self.classification,
            'similarity_score': self.similarity_score,
            'reason': self.reason,
            'detailed_scores': self.detailed_scores,
            'timestamp': self.timestamp,
            'contract_text': self.contract_text[:200],  # Truncate for display
            'template_text': self.template_text[:200]
        }


@dataclass
class ContractClassification:
    """Container for full contract classification"""
    contract_id: str
    state: str
    results: List[ClassificationResult] = field(default_factory=list)
    standard_count: int = 0
    non_standard_count: int = 0
    processing_time: float = 0.0
    
    def add_result(self, result: ClassificationResult):
        """Add classification result"""
        self.results.append(result)
        if result.classification == "Standard":
            self.standard_count += 1
        else:
            self.non_standard_count += 1
            
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'contract_id': self.contract_id,
            'state': self.state,
            'results': [r.to_dict() for r in self.results],
            'standard_count': self.standard_count,
            'non_standard_count': self.non_standard_count,
            'total_attributes': len(self.results),
            'processing_time': self.processing_time
        }
        
    def get_summary(self) -> str:
        """Get summary string"""
        return (
            f"Contract {self.contract_id} ({self.state}): "
            f"{self.standard_count} Standard, {self.non_standard_count} Non-Standard"
        )


class ClauseClassifier:
    """Main classification logic"""
    
    def __init__(self, comparator: TextComparator, config: Dict):
        """
        Initialize classifier
        
        Args:
            comparator: Text comparator instance
            config: Classification configuration
        """
        self.comparator = comparator
        self.config = config
        self.thresholds = config.get('thresholds', {})
        self.standard_threshold = self.thresholds.get('standard', 0.8)
        self.critical_threshold = self.thresholds.get('critical_threshold', 0.6)
        self.critical_attributes = self.thresholds.get('critical_attributes', [])
        
    def classify_clause(
        self,
        clause: ClauseData,
        standard_text: str
    ) -> ClassificationResult:
        """
        Classify a single clause
        
        Args:
            clause: Extracted clause data
            standard_text: Standard template text
            
        Returns:
            Classification result
        """
        logger.info(f"Classifying {clause.attribute_name}")
        
        # Get similarity scores
        scores = self.comparator.get_detailed_scores(
            clause.extracted_text,
            standard_text
        )
        
        # Apply business rules
        business_rule_reason = self.comparator.apply_business_rules(
            clause.extracted_text,
            standard_text
        )
        
        # Determine threshold based on attribute criticality
        threshold = self._get_threshold(clause.attribute_name)
        
        # Determine classification
        classification, reason = self._determine_classification(
            scores,
            threshold,
            business_rule_reason
        )
        
        return ClassificationResult(
            attribute_name=clause.attribute_name,
            classification=classification,
            similarity_score=scores.weighted_score,
            reason=reason,
            detailed_scores=scores.to_dict(),
            contract_text=clause.extracted_text,
            template_text=standard_text
        )
        
    def classify_contract(
        self,
        contract_id: str,
        state: str,
        extracted_clauses: Dict[str, ClauseData],
        standard_clauses: Dict[str, str]
    ) -> ContractClassification:
        """
        Classify all clauses in a contract
        
        Args:
            contract_id: Contract identifier
            state: State code
            extracted_clauses: Extracted clauses from contract
            standard_clauses: Standard template clauses
            
        Returns:
            Complete contract classification
        """
        classification = ContractClassification(
            contract_id=contract_id,
            state=state
        )
        
        start_time = datetime.now()
        
        for attribute_name, clause in extracted_clauses.items():
            if attribute_name in standard_clauses:
                result = self.classify_clause(
                    clause,
                    standard_clauses[attribute_name]
                )
                classification.add_result(result)
            else:
                # No standard clause found - mark as non-standard
                result = ClassificationResult(
                    attribute_name=attribute_name,
                    classification="Non-Standard",
                    similarity_score=0.0,
                    reason="No standard template clause found",
                    contract_text=clause.extracted_text,
                    template_text=""
                )
                classification.add_result(result)
                
        # Handle missing attributes (in template but not extracted)
        for attribute_name in standard_clauses:
            if attribute_name not in extracted_clauses:
                result = ClassificationResult(
                    attribute_name=attribute_name,
                    classification="Non-Standard",
                    similarity_score=0.0,
                    reason="Attribute not found in contract",
                    contract_text="",
                    template_text=standard_clauses[attribute_name]
                )
                classification.add_result(result)
                
        classification.processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(classification.get_summary())
        
        return classification
        
    def _get_threshold(self, attribute_name: str) -> float:
        """
        Get classification threshold for attribute
        
        Args:
            attribute_name: Name of attribute
            
        Returns:
            Threshold value
        """
        if attribute_name in self.critical_attributes:
            return self.critical_threshold
        return self.standard_threshold
        
    def _determine_classification(
        self,
        scores: SimilarityScores,
        threshold: float,
        business_rule_reason: str
    ) -> tuple[str, str]:
        """
        Determine final classification
        
        Args:
            scores: Similarity scores
            threshold: Classification threshold
            business_rule_reason: Business rule violation reason
            
        Returns:
            Tuple of (classification, reason)
        """
        # Check business rules first
        if business_rule_reason:
            return "Non-Standard", business_rule_reason
            
        # Check exact match
        if scores.exact_match == 1.0:
            return "Standard", "Exact match with template"
            
        # Check weighted score
        if scores.weighted_score >= threshold:
            # Additional checks for high similarity
            if scores.semantic_score >= 0.9:
                return "Standard", f"High semantic similarity ({scores.semantic_score:.2%})"
            elif scores.fuzzy_score >= 0.85:
                return "Standard", f"High fuzzy similarity ({scores.fuzzy_score:.2%})"
            else:
                return "Standard", f"Meets threshold ({scores.weighted_score:.2%} >= {threshold:.2%})"
        else:
            # Determine specific reason for non-standard
            if scores.semantic_score < 0.5:
                return "Non-Standard", "Low semantic similarity - different meaning"
            elif scores.fuzzy_score < 0.5:
                return "Non-Standard", "Low text similarity - different wording"
            else:
                return "Non-Standard", f"Below threshold ({scores.weighted_score:.2%} < {threshold:.2%})"


class ClassificationModelFactory:
    """Factory for creating classification models"""
    
    @staticmethod
    def create_classifier(model_type: str, config: Dict) -> ClauseClassifier:
        """
        Create classifier based on model type
        
        Args:
            model_type: Type of classifier
            config: Configuration dictionary
            
        Returns:
            Configured classifier instance
        """
        # Create appropriate comparator based on model type
        if model_type == "hybrid":
            comparator_config = {
                'semantic': config.get('semantic', {}),
                'fuzzy': config.get('fuzzy', {}),
                'exact': config.get('exact', {})
            }
        elif model_type == "semantic":
            comparator_config = {
                'semantic': {'weight': 1.0, 'enabled': True},
                'fuzzy': {'weight': 0.0},
                'exact': {'weight': 0.0}
            }
        elif model_type == "fuzzy":
            comparator_config = {
                'semantic': {'weight': 0.0, 'enabled': False},
                'fuzzy': {'weight': 0.7},
                'exact': {'weight': 0.3}
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        comparator = TextComparator(comparator_config)
        return ClauseClassifier(comparator, config)


class BatchClassifier:
    """Handle batch classification of multiple contracts"""
    
    def __init__(self, classifier: ClauseClassifier):
        """
        Initialize batch classifier
        
        Args:
            classifier: Single contract classifier
        """
        self.classifier = classifier
        
    def classify_batch(
        self,
        contracts: List[Dict],
        templates: Dict[str, Dict[str, str]]
    ) -> List[ContractClassification]:
        """
        Classify multiple contracts
        
        Args:
            contracts: List of contract data dictionaries
            templates: Templates by state
            
        Returns:
            List of classification results
        """
        results = []
        
        for contract in contracts:
            state = contract['state']
            if state not in templates:
                logger.error(f"No template found for state {state}")
                continue
                
            classification = self.classifier.classify_contract(
                contract['id'],
                state,
                contract['extracted_clauses'],
                templates[state]
            )
            results.append(classification)
            
        return results