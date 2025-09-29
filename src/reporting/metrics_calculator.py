"""
Metrics calculation module for classification results
"""

from typing import Dict, List
import numpy as np
from collections import defaultdict

from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


class MetricsCalculator:
    """Calculate metrics from classification results"""
    
    def calculate_summary(self, classifications: List) -> Dict:
        """
        Calculate summary metrics
        
        Args:
            classifications: List of ContractClassification objects
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'summary': self._calculate_overall_summary(classifications),
            'by_attribute': self._calculate_by_attribute(classifications),
            'by_state': self._calculate_by_state(classifications),
            'risk_contracts': self._identify_risk_contracts(classifications),
            'confidence_distribution': self._calculate_confidence_distribution(classifications)
        }
        
        logger.info(f"Calculated metrics for {len(classifications)} contracts")
        return metrics
        
    def _calculate_overall_summary(self, classifications: List) -> Dict:
        """Calculate overall summary metrics"""
        total_contracts = len(classifications)
        total_clauses = sum(len(c.results) for c in classifications)
        standard_clauses = sum(c.standard_count for c in classifications)
        non_standard_clauses = sum(c.non_standard_count for c in classifications)
        
        # Avoid division by zero
        if total_clauses == 0:
            standard_percentage = 0
            non_standard_percentage = 0
        else:
            standard_percentage = (standard_clauses / total_clauses) * 100
            non_standard_percentage = (non_standard_clauses / total_clauses) * 100
        
        # Average confidence
        all_scores = []
        for contract in classifications:
            for result in contract.results:
                all_scores.append(result.similarity_score)
        
        avg_confidence = np.mean(all_scores) if all_scores else 0
        
        # Contracts with issues
        contracts_with_issues = sum(
            1 for c in classifications
            if c.non_standard_count > 0
        )
        
        return {
            'total_contracts': total_contracts,
            'total_clauses': total_clauses,
            'standard_clauses': standard_clauses,
            'non_standard_clauses': non_standard_clauses,
            'standard_percentage': standard_percentage,
            'non_standard_percentage': non_standard_percentage,
            'avg_confidence': avg_confidence,
            'contracts_with_issues': contracts_with_issues,
            'issue_rate': (contracts_with_issues / total_contracts * 100) if total_contracts > 0 else 0
        }
        
    def _calculate_by_attribute(self, classifications: List) -> Dict:
        """Calculate metrics grouped by attribute"""
        attribute_metrics = defaultdict(lambda: {
            'total': 0,
            'standard': 0,
            'non_standard': 0,
            'confidence_scores': []
        })
        
        for contract in classifications:
            for result in contract.results:
                attr_name = result.attribute_name
                attribute_metrics[attr_name]['total'] += 1
                
                if result.classification == "Standard":
                    attribute_metrics[attr_name]['standard'] += 1
                else:
                    attribute_metrics[attr_name]['non_standard'] += 1
                    
                attribute_metrics[attr_name]['confidence_scores'].append(
                    result.similarity_score
                )
        
        # Calculate averages
        for attr_name in attribute_metrics:
            scores = attribute_metrics[attr_name]['confidence_scores']
            attribute_metrics[attr_name]['avg_confidence'] = np.mean(scores) if scores else 0
            attribute_metrics[attr_name]['min_confidence'] = np.min(scores) if scores else 0
            attribute_metrics[attr_name]['max_confidence'] = np.max(scores) if scores else 0
            # Remove the raw scores list for cleaner output
            del attribute_metrics[attr_name]['confidence_scores']
        
        return dict(attribute_metrics)
        
    def _calculate_by_state(self, classifications: List) -> Dict:
        """Calculate metrics grouped by state"""
        state_metrics = defaultdict(lambda: {
            'contracts': 0,
            'standard': 0,
            'non_standard': 0,
            'total_clauses': 0
        })
        
        for contract in classifications:
            state = contract.state
            state_metrics[state]['contracts'] += 1
            state_metrics[state]['standard'] += contract.standard_count
            state_metrics[state]['non_standard'] += contract.non_standard_count
            state_metrics[state]['total_clauses'] += len(contract.results)
        
        # Calculate percentages
        for state in state_metrics:
            total = state_metrics[state]['total_clauses']
            if total > 0:
                state_metrics[state]['standard_percentage'] = (
                    state_metrics[state]['standard'] / total * 100
                )
                state_metrics[state]['non_standard_percentage'] = (
                    state_metrics[state]['non_standard'] / total * 100
                )
            else:
                state_metrics[state]['standard_percentage'] = 0
                state_metrics[state]['non_standard_percentage'] = 0
        
        return dict(state_metrics)
        
    def _identify_risk_contracts(self, classifications: List) -> List[Dict]:
        """Identify high-risk contracts"""
        risk_contracts = []
        
        for contract in classifications:
            risk_score = 0
            risk_factors = []
            
            # Check for non-standard critical attributes
            critical_attributes = ['Medicaid Timely Filing', 'Medicare Timely Filing']
            
            for result in contract.results:
                if result.classification == "Non-Standard":
                    if result.attribute_name in critical_attributes:
                        risk_score += 3
                        risk_factors.append(f"Critical: {result.attribute_name}")
                    else:
                        risk_score += 1
                        risk_factors.append(result.attribute_name)
                    
                # Low confidence even if standard
                if result.similarity_score < 0.6:
                    risk_score += 1
                    risk_factors.append(f"Low confidence: {result.attribute_name}")
            
            # High percentage of non-standard clauses
            if len(contract.results) > 0:
                non_standard_ratio = contract.non_standard_count / len(contract.results)
                if non_standard_ratio > 0.4:
                    risk_score += 2
                    risk_factors.append(f"High non-standard ratio: {non_standard_ratio:.1%}")
            
            if risk_score >= 3:  # Threshold for high risk
                risk_contracts.append({
                    'contract_id': contract.contract_id,
                    'state': contract.state,
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'non_standard_count': contract.non_standard_count,
                    'total_attributes': len(contract.results)
                })
        
        # Sort by risk score
        risk_contracts.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return risk_contracts
        
    def _calculate_confidence_distribution(self, classifications: List) -> Dict:
        """Calculate confidence score distribution"""
        all_scores = []
        
        for contract in classifications:
            for result in contract.results:
                all_scores.append(result.similarity_score)
        
        if not all_scores:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'quartiles': [0, 0, 0]
            }
        
        return {
            'mean': np.mean(all_scores),
            'median': np.median(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'quartiles': np.percentile(all_scores, [25, 50, 75]).tolist(),
            'bins': self._create_histogram_bins(all_scores)
        }
        
    def _create_histogram_bins(self, scores: List[float], n_bins: int = 10) -> Dict:
        """Create histogram bins for score distribution"""
        hist, bin_edges = np.histogram(scores, bins=n_bins)
        
        bins = {}
        for i in range(len(hist)):
            bin_label = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            bins[bin_label] = int(hist[i])
        
        return bins
        
    def calculate_accuracy_metrics(self, classifications: List, ground_truth: Dict = None) -> Dict:
        """
        Calculate accuracy metrics if ground truth is available
        
        Args:
            classifications: List of classification results
            ground_truth: Dictionary of expected classifications
            
        Returns:
            Accuracy metrics
        """
        if not ground_truth:
            logger.warning("No ground truth provided for accuracy calculation")
            return {}
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        for contract in classifications:
            for result in contract.results:
                key = f"{contract.contract_id}_{result.attribute_name}"
                
                if key in ground_truth:
                    expected = ground_truth[key]
                    predicted = result.classification
                    
                    if predicted == "Standard" and expected == "Standard":
                        true_positive += 1
                    elif predicted == "Non-Standard" and expected == "Non-Standard":
                        true_negative += 1
                    elif predicted == "Standard" and expected == "Non-Standard":
                        false_positive += 1
                    elif predicted == "Non-Standard" and expected == "Standard":
                        false_negative += 1
        
        total = true_positive + true_negative + false_positive + false_negative
        
        if total == 0:
            return {}
        
        accuracy = (true_positive + true_negative) / total
        
        # Precision for "Standard" class
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        
        # Recall for "Standard" class
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'true_positive': true_positive,
                'true_negative': true_negative,
                'false_positive': false_positive,
                'false_negative': false_negative
            }
        }
        
    def generate_performance_report(self, classifications: List) -> Dict:
        """Generate performance metrics for the classification system"""
        
        # Processing time statistics
        processing_times = [c.processing_time for c in classifications if hasattr(c, 'processing_time')]
        
        performance_metrics = {
            'processing': {
                'total_contracts': len(classifications),
                'avg_time_per_contract': np.mean(processing_times) if processing_times else 0,
                'min_time': np.min(processing_times) if processing_times else 0,
                'max_time': np.max(processing_times) if processing_times else 0,
                'total_time': np.sum(processing_times) if processing_times else 0
            }
        }
        
        # Extraction success rate
        total_expected_attributes = len(classifications) * 5  # 5 attributes per contract
        total_extracted = sum(len(c.results) for c in classifications)
        
        performance_metrics['extraction'] = {
            'expected_attributes': total_expected_attributes,
            'extracted_attributes': total_extracted,
            'extraction_rate': (total_extracted / total_expected_attributes * 100) if total_expected_attributes > 0 else 0
        }
        
        # Classification confidence
        confidence_scores = []
        for contract in classifications:
            for result in contract.results:
                confidence_scores.append(result.similarity_score)
        
        if confidence_scores:
            performance_metrics['confidence'] = {
                'high_confidence': sum(1 for s in confidence_scores if s >= 0.8) / len(confidence_scores) * 100,
                'medium_confidence': sum(1 for s in confidence_scores if 0.6 <= s < 0.8) / len(confidence_scores) * 100,
                'low_confidence': sum(1 for s in confidence_scores if s < 0.6) / len(confidence_scores) * 100
            }
        
        return performance_metrics