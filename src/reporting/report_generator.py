"""
Report generation module for classification results
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime

from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


class ReportGenerator:
    """Generate various report formats"""
    
    def __init__(self, output_manager):
        """
        Initialize report generator
        
        Args:
            output_manager: Output manager instance
        """
        self.output_manager = output_manager
        
    def generate_all_reports(
        self,
        classifications: List,
        metrics: Dict
    ) -> Dict[str, Path]:
        """
        Generate all report types
        
        Args:
            classifications: List of classification results
            metrics: Calculated metrics
            
        Returns:
            Dictionary of report type to file path
        """
        report_paths = {}
        
        # Generate CSV report
        report_paths['csv'] = self.generate_csv_report(classifications)
        
        # Generate JSON report
        report_paths['json'] = self.generate_json_report(classifications)
        
        # Generate Excel report
        report_paths['excel'] = self.generate_excel_report(classifications, metrics)
        
        # Generate summary report
        report_paths['summary'] = self.generate_summary_report(classifications, metrics)
        
        logger.info(f"Generated {len(report_paths)} reports")
        
        return report_paths
        
    def generate_csv_report(self, classifications: List) -> Path:
        """
        Generate CSV report
        
        Args:
            classifications: List of classification results
            
        Returns:
            Path to generated CSV file
        """
        csv_path = self.output_manager.get_output_path('classification_results.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Contract_ID', 'State', 'Attribute', 'Classification',
                'Similarity_Score', 'Reason', 'Contract_Text', 'Template_Text',
                'Timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for contract in classifications:
                for result in contract.results:
                    writer.writerow({
                        'Contract_ID': contract.contract_id,
                        'State': contract.state,
                        'Attribute': result.attribute_name,
                        'Classification': result.classification,
                        'Similarity_Score': f"{result.similarity_score:.4f}",
                        'Reason': result.reason,
                        'Contract_Text': result.contract_text[:500],  # Truncate
                        'Template_Text': result.template_text[:500],  # Truncate
                        'Timestamp': result.timestamp
                    })
        
        logger.info(f"CSV report saved to {csv_path}")
        return csv_path
        
    def generate_json_report(self, classifications: List) -> Path:
        """
        Generate JSON report
        
        Args:
            classifications: List of classification results
            
        Returns:
            Path to generated JSON file
        """
        json_data = []
        
        for contract in classifications:
            contract_data = contract.to_dict()
            json_data.append(contract_data)
        
        json_path = self.output_manager.save_json(json_data, 'classification_results.json')
        
        logger.info(f"JSON report saved to {json_path}")
        return json_path
        
    def generate_excel_report(self, classifications: List, metrics: Dict) -> Path:
        """
        Generate Excel report with multiple sheets
        
        Args:
            classifications: List of classification results
            metrics: Calculated metrics
            
        Returns:
            Path to generated Excel file
        """
        excel_path = self.output_manager.get_output_path('classification_results.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = pd.DataFrame([metrics['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Contract Overview
            contract_data = []
            for contract in classifications:
                contract_data.append({
                    'Contract ID': contract.contract_id,
                    'State': contract.state,
                    'Standard Count': contract.standard_count,
                    'Non-Standard Count': contract.non_standard_count,
                    'Total Attributes': len(contract.results),
                    'Processing Time (s)': contract.processing_time
                })
            
            contracts_df = pd.DataFrame(contract_data)
            contracts_df.to_excel(writer, sheet_name='Contracts', index=False)
            
            # Sheet 3: Detailed Classifications
            detailed_data = []
            for contract in classifications:
                for result in contract.results:
                    detailed_data.append({
                        'Contract ID': contract.contract_id,
                        'State': contract.state,
                        'Attribute': result.attribute_name,
                        'Classification': result.classification,
                        'Similarity Score': result.similarity_score,
                        'Exact Match': result.detailed_scores.get('exact_match', 0),
                        'Fuzzy Score': result.detailed_scores.get('fuzzy_score', 0),
                        'Semantic Score': result.detailed_scores.get('semantic_score', 0),
                        'Reason': result.reason
                    })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Sheet 4: Attribute Analysis
            if 'by_attribute' in metrics:
                attribute_df = pd.DataFrame(metrics['by_attribute']).T
                attribute_df.to_excel(writer, sheet_name='By Attribute')
            
            # Sheet 5: State Analysis
            if 'by_state' in metrics:
                state_df = pd.DataFrame(metrics['by_state']).T
                state_df.to_excel(writer, sheet_name='By State')
        
        logger.info(f"Excel report saved to {excel_path}")
        return excel_path
        
    def generate_summary_report(self, classifications: List, metrics: Dict) -> Path:
        """
        Generate human-readable summary report
        
        Args:
            classifications: List of classification results
            metrics: Calculated metrics
            
        Returns:
            Path to generated summary file
        """
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("HILABS CONTRACT CLASSIFICATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*40)
        report_lines.append(f"Total Contracts Processed: {metrics['summary']['total_contracts']}")
        report_lines.append(f"Total Clauses Analyzed: {metrics['summary']['total_clauses']}")
        report_lines.append(f"Standard Clauses: {metrics['summary']['standard_clauses']} ({metrics['summary']['standard_percentage']:.1f}%)")
        report_lines.append(f"Non-Standard Clauses: {metrics['summary']['non_standard_clauses']} ({metrics['summary']['non_standard_percentage']:.1f}%)")
        report_lines.append("")
        
        # Contracts with Non-Standard Clauses
        report_lines.append("CONTRACTS REQUIRING REVIEW")
        report_lines.append("-"*40)
        
        contracts_with_issues = [
            c for c in classifications
            if c.non_standard_count > 0
        ]
        
        if contracts_with_issues:
            for contract in contracts_with_issues:
                report_lines.append(f"• {contract.contract_id} ({contract.state}): {contract.non_standard_count} non-standard clause(s)")
                
                # List non-standard attributes
                non_standard = [
                    r for r in contract.results
                    if r.classification == "Non-Standard"
                ]
                for result in non_standard[:3]:  # Show top 3
                    report_lines.append(f"  - {result.attribute_name}: {result.reason}")
                    
                if len(non_standard) > 3:
                    report_lines.append(f"  ... and {len(non_standard) - 3} more")
                    
        else:
            report_lines.append("No contracts with non-standard clauses found.")
        
        report_lines.append("")
        
        # Classification by Attribute
        report_lines.append("CLASSIFICATION BY ATTRIBUTE")
        report_lines.append("-"*40)
        
        if 'by_attribute' in metrics:
            for attr_name, attr_metrics in metrics['by_attribute'].items():
                total = attr_metrics['total']
                standard = attr_metrics['standard']
                non_standard = attr_metrics['non_standard']
                avg_confidence = attr_metrics['avg_confidence']
                
                report_lines.append(f"{attr_name}:")
                report_lines.append(f"  Standard: {standard}/{total} ({standard/total*100:.1f}%)")
                report_lines.append(f"  Non-Standard: {non_standard}/{total} ({non_standard/total*100:.1f}%)")
                report_lines.append(f"  Avg Confidence: {avg_confidence:.2%}")
                report_lines.append("")
        
        # Classification by State
        report_lines.append("CLASSIFICATION BY STATE")
        report_lines.append("-"*40)
        
        if 'by_state' in metrics:
            for state, state_metrics in metrics['by_state'].items():
                report_lines.append(f"{state}:")
                report_lines.append(f"  Contracts: {state_metrics['contracts']}")
                report_lines.append(f"  Standard Clauses: {state_metrics['standard']}")
                report_lines.append(f"  Non-Standard Clauses: {state_metrics['non_standard']}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-"*40)
        
        if metrics['summary']['non_standard_percentage'] > 20:
            report_lines.append("⚠ High percentage of non-standard clauses detected.")
            report_lines.append("  Recommend thorough review of contract templates and negotiation strategies.")
        
        # Attributes with low confidence
        low_confidence_attrs = []
        if 'by_attribute' in metrics:
            for attr_name, attr_metrics in metrics['by_attribute'].items():
                if attr_metrics['avg_confidence'] < 0.7:
                    low_confidence_attrs.append(attr_name)
        
        if low_confidence_attrs:
            report_lines.append(f"⚠ Low confidence scores for: {', '.join(low_confidence_attrs)}")
            report_lines.append("  Consider manual review of these attributes.")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save report
        report_content = '\n'.join(report_lines)
        summary_path = self.output_manager.save_text(report_content, 'summary_report.txt')
        
        logger.info(f"Summary report saved to {summary_path}")
        return summary_path