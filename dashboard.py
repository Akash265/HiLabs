"""
Streamlit dashboard for HiLabs Contract Classification System - Optimized with Full Features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import yaml
from datetime import datetime
import time
import difflib
import sys
import os
import hashlib
from functools import lru_cache

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="HiLabs Contract Classification",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import with error handling
try:
    from main import ContractProcessor
    from src.utils.file_handler import FileHandler, OutputManager
    from src.data_processing.pdf_parser import PDFParser
    from src.data_processing.clause_extractor import ClauseExtractor
    from src.classification.clause_classifier import ClassificationModelFactory, ContractClassification, ClassificationResult
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .standard {
        color: #28a745;
        font-weight: bold;
    }
    .non-standard {
        color: #dc3545;
        font-weight: bold;
    }
    .confidence-high {
        background-color: #d4edda;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .confidence-medium {
        background-color: #fff3cd;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .confidence-low {
        background-color: #f8d7da;
        padding: 2px 6px;
        border-radius: 3px;
    }
    .diff-highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: bold;
    }
    .matching-text {
        background-color: #c8e6c9;
        padding: 2px 4px;
        border-radius: 2px;
    }
    .missing-text {
        background-color: #ffcdd2;
        padding: 2px 4px;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with caching
if 'processed_contracts' not in st.session_state:
    st.session_state.processed_contracts = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = 'fuzzy'
if 'pdf_cache' not in st.session_state:
    st.session_state.pdf_cache = {}
if 'template_cache' not in st.session_state:
    st.session_state.template_cache = {}
if 'config_cache' not in st.session_state:
    st.session_state.config_cache = None


@lru_cache(maxsize=1)
def load_config_cached():
    """Cache configuration loading"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def reset_processor():
    """Reset the processor to allow new model configuration"""
    if 'processor' in st.session_state:
        del st.session_state.processor
    st.session_state.processor = None
    st.session_state.processing = False


def update_model_config(model_type: str, threshold: float):
    """Update configuration with selected model type"""
    try:
        # Check if semantic models are available
        semantic_available = False
        try:
            import sentence_transformers
            semantic_available = True
        except ImportError:
            pass
        
        # If semantic/hybrid selected but not available, warn user
        if model_type in ["semantic", "hybrid"] and not semantic_available:
            st.warning("⚠️ Semantic models not available. Using fuzzy mode instead.")
            st.info("To use semantic models, install: pip install sentence-transformers transformers torch")
            model_type = "fuzzy"
        
        # Load current config
        config = load_config_cached()
        
        # Update model type
        config['classification']['model_type'] = model_type
        config['classification']['thresholds']['standard'] = threshold
        config['classification']['thresholds']['non_standard'] = threshold
        
        # Adjust weights based on model type
        if model_type == "fuzzy":
            config['classification']['semantic']['enabled'] = False
            config['classification']['semantic']['weight'] = 0.0
            config['classification']['fuzzy']['weight'] = 0.7
            config['classification']['exact']['weight'] = 0.3
        elif model_type == "semantic":
            config['classification']['semantic']['enabled'] = True
            config['classification']['semantic']['weight'] = 0.8
            config['classification']['fuzzy']['weight'] = 0.1
            config['classification']['exact']['weight'] = 0.1
        else:  # hybrid
            config['classification']['semantic']['enabled'] = True
            config['classification']['semantic']['weight'] = 0.5
            config['classification']['fuzzy']['weight'] = 0.3
            config['classification']['exact']['weight'] = 0.2
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.success(f"✅ Configuration updated to use {model_type} model with threshold {threshold}")
        
        # Store the current model type
        st.session_state.current_model_type = model_type
        
        # Clear caches
        st.session_state.current_results = None
        st.session_state.processed_contracts = []
        st.session_state.config_cache = config
        load_config_cached.cache_clear()
        
        # Reset processor to use new config
        reset_processor()
        
    except Exception as e:
        st.error(f"Error updating configuration: {e}")
        st.info("Using fuzzy mode as fallback")


def load_configuration():
    """Load configuration file with caching"""
    if st.session_state.config_cache is None:
        st.session_state.config_cache = load_config_cached()
    return st.session_state.config_cache


def initialize_processor():
    """Initialize the contract processor with caching"""
    # Check if model changed
    current_model = st.session_state.get('current_model_type', 'fuzzy')
    config_model = load_configuration().get('classification', {}).get('model_type', 'fuzzy')
    
    # Force re-initialization if model changed
    if st.session_state.processor is not None and current_model != config_model:
        st.session_state.processor = None
    
    if st.session_state.processor is None:
        with st.spinner("Initializing processor..."):
            try:
                processor = DynamicContractProcessor()
                st.session_state.processor = processor
            except Exception as e:
                st.error(f"Error initializing processor: {e}")
                st.info("Try selecting 'fuzzy' model for faster initialization")
                return None
    return st.session_state.processor


class DynamicContractProcessor:
    """Contract processor with dynamic model switching and caching"""
    
    def __init__(self):
        """Initialize with current config"""
        self.config = load_configuration()
        self._classifiers = {}  # Cache classifiers
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize components once"""
        # Initialize file handlers
        self.file_handler = FileHandler(
            self.config['paths']['contracts_dir'],
            self.config['paths']['templates_dir']
        )
        self.output_manager = OutputManager(self.config['paths']['output_dir'])
        
        # Initialize PDF parser
        self.pdf_parser = PDFParser(self.config.get('pdf_extraction', {}))
        
        # Initialize clause extractor
        self.clause_extractor = ClauseExtractor(self.config['attributes'])
        
        # Initialize classifier
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize classifier based on selected model with caching"""
        model_type = self.config['classification']['model_type']
        
        # Check if we already have this classifier
        if model_type in self._classifiers:
            self.classifier = self._classifiers[model_type]
            self.classification_method = 'fast' if model_type == 'fuzzy' else 'full'
            return
        
        # Create new classifier
        if model_type == "fuzzy":
            try:
                from src.classification.fast_classifier import FastClassifier
                self.classifier = FastClassifier(
                    threshold=self.config['classification']['thresholds']['standard']
                )
                self.classification_method = 'fast'
            except ImportError:
                self.classifier = ClassificationModelFactory.create_classifier(
                    'fuzzy',
                    self.config['classification']
                )
                self.classification_method = 'full'
        else:
            self.classifier = ClassificationModelFactory.create_classifier(
                model_type,
                self.config['classification']
            )
            self.classification_method = 'full'
        
        # Cache the classifier
        self._classifiers[model_type] = self.classifier
    
    def process_template(self, template_path):
        """Process template file with caching"""
        template_key = str(template_path)
        
        # Check cache
        if template_key in st.session_state.template_cache:
            return st.session_state.template_cache[template_key]
        
        # Process template
        text = self.pdf_parser.parse_pdf(template_path)
        sections = self.pdf_parser.extract_sections(text)
        extracted_clauses = self.clause_extractor.extract_all_clauses(text, sections)
        
        standard_clauses = {
            name: clause.extracted_text
            for name, clause in extracted_clauses.items()
        }
        
        # Cache results
        st.session_state.template_cache[template_key] = standard_clauses
        return standard_clauses
    
    def process_contract(self, contract_path, template_clauses, state):
        """Process single contract with caching"""
        contract_id = contract_path.stem
        
        # Generate cache key
        file_size = contract_path.stat().st_size if contract_path.exists() else 0
        cache_key = f"{contract_id}_{file_size}_{state}"
        
        # Check PDF cache
        if cache_key in st.session_state.pdf_cache:
            text, sections, extracted_clauses = st.session_state.pdf_cache[cache_key]
        else:
            # Extract text and clauses
            text = self.pdf_parser.parse_pdf(contract_path)
            sections = self.pdf_parser.extract_sections(text)
            extracted_clauses = self.clause_extractor.extract_all_clauses(text, sections)
            
            # Cache results
            st.session_state.pdf_cache[cache_key] = (text, sections, extracted_clauses)
        
        # Get current model type from config (not session state)
        current_model_type = self.config['classification']['model_type']
        
        if self.classification_method == 'fast':
            # Fast classification path
            contract_data = {
                'id': contract_id,
                'clauses': {
                    name: clause.extracted_text
                    for name, clause in extracted_clauses.items()
                }
            }
            results = self.classifier.classify_batch([contract_data], template_clauses)
            
            # Convert to standard format
            classification = ContractClassification(contract_id, state)
            classification.model_type_used = current_model_type
            
            if results:
                for r in results[0]['results']:
                    # Calculate detailed scores
                    contract_text = contract_data['clauses'].get(r['attribute'], '')
                    template_text = template_clauses.get(r['attribute'], '')
                    
                    detailed_scores = self._calculate_detailed_scores(
                        contract_text, template_text, current_model_type
                    )
                    
                    # For fuzzy mode, the weighted score should just be the fuzzy score
                    if current_model_type == 'fuzzy':
                        weighted_score = r['score']  # Use fuzzy score directly
                    else:
                        weighted_score = r['score']
                    
                    result = ClassificationResult(
                        attribute_name=r['attribute'],
                        classification=r['classification'],
                        similarity_score=weighted_score,
                        reason=self._generate_detailed_reason(r, contract_data['clauses'], template_clauses),
                        contract_text=contract_text,
                        template_text=template_text,
                        matching_phrases=r.get('matching_phrases', [])
                    )
                    result.detailed_scores = detailed_scores
                    result.weighted_score = weighted_score
                    classification.add_result(result)
            
            return classification
        else:
            # Full classification
            classification = self.classifier.classify_contract(
                contract_id,
                state,
                extracted_clauses,
                template_clauses
            )
            
            classification.model_type_used = current_model_type
            
            # Ensure detailed scores are properly set and calculate weighted score
            for result in classification.results:
                if not hasattr(result, 'detailed_scores') or not result.detailed_scores:
                    result.detailed_scores = self._calculate_detailed_scores(
                        result.contract_text,
                        result.template_text,
                        current_model_type
                    )
                
                # Ensure model type is stored
                if result.detailed_scores:
                    result.detailed_scores['model_type'] = current_model_type
                
                # Calculate proper weighted score
                if current_model_type == 'hybrid' or current_model_type == 'semantic':
                    weighted_score = self._calculate_weighted_score(result.detailed_scores, current_model_type)
                    result.weighted_score = weighted_score
                    # Update the similarity score to be the weighted score
                    result.similarity_score = weighted_score
            
            return classification
    
    def _calculate_weighted_score(self, detailed_scores, model_type):
        """Calculate proper weighted score based on model configuration"""
        if not detailed_scores:
            return 0.0
        
        # Get weights from config
        weights = self.config['classification']
        
        fuzzy_weight = weights.get('fuzzy', {}).get('weight', 0.0)
        exact_weight = weights.get('exact', {}).get('weight', 0.0)
        semantic_weight = weights.get('semantic', {}).get('weight', 0.0)
        
        # Get individual scores
        fuzzy_score = detailed_scores.get('fuzzy_score', 0.0)
        exact_score = detailed_scores.get('exact_score', 0.0)
        semantic_score = detailed_scores.get('semantic_score', 0.0)
        
        # Calculate weighted sum
        if model_type == 'fuzzy':
            # For fuzzy mode, use only fuzzy and exact scores
            total_weight = fuzzy_weight + exact_weight
            if total_weight > 0:
                weighted = (fuzzy_score * fuzzy_weight + exact_score * exact_weight) / total_weight
            else:
                weighted = fuzzy_score
        elif model_type == 'semantic':
            # For semantic mode, primarily use semantic score
            total_weight = fuzzy_weight + exact_weight + semantic_weight
            if total_weight > 0:
                weighted = (fuzzy_score * fuzzy_weight + 
                          exact_score * exact_weight + 
                          semantic_score * semantic_weight) / total_weight
            else:
                weighted = semantic_score
        else:  # hybrid
            # For hybrid mode, use all three scores
            total_weight = fuzzy_weight + exact_weight + semantic_weight
            if total_weight > 0:
                weighted = (fuzzy_score * fuzzy_weight + 
                          exact_score * exact_weight + 
                          semantic_score * semantic_weight) / total_weight
            else:
                weighted = (fuzzy_score + exact_score + semantic_score) / 3
        
        return weighted
    
    def _calculate_detailed_scores(self, contract_text, template_text, model_type):
        """Calculate detailed scores for classification"""
        scores = {'model_type': model_type}
        
        if contract_text and template_text:
            # Fuzzy score
            from difflib import SequenceMatcher
            fuzzy_score = SequenceMatcher(None, 
                                         contract_text.lower(), 
                                         template_text.lower()).ratio()
            scores['fuzzy_score'] = fuzzy_score
            
            # Exact match score
            contract_words = set(contract_text.lower().split())
            template_words = set(template_text.lower().split())
            if template_words:
                exact_score = len(contract_words & template_words) / len(template_words)
                scores['exact_score'] = exact_score
            else:
                scores['exact_score'] = 0
        else:
            scores['fuzzy_score'] = 0
            scores['exact_score'] = 0
        
        # Semantic score placeholder
        if model_type in ['semantic', 'hybrid']:
            scores['semantic_score'] = scores.get('semantic_score', 0)
        else:
            scores['semantic_score'] = 0
        
        return scores
    
    def _generate_detailed_reason(self, result, contract_clauses, template_clauses):
        """Generate detailed reasoning for classification decision"""
        attr = result['attribute']
        score = result.get('score', 0)
        classification = result['classification']
        
        contract_text = contract_clauses.get(attr, '')
        template_text = template_clauses.get(attr, '')
        
        if classification == "Not Found":
            return f"The '{attr}' clause was not found in the contract. This attribute is expected but missing."
        
        model_type = self.config.get('classification', {}).get('model_type', 'fuzzy')
        model_context = f" (using {model_type} model)"
        
        if contract_text and template_text:
            matcher = difflib.SequenceMatcher(None, contract_text.lower(), template_text.lower())
            ratio = matcher.ratio()
            
            if classification == "Standard":
                if ratio > 0.95:
                    return f"Near-exact match with template (similarity: {score:.1%}{model_context}). The contract clause closely follows the standard template language."
                elif ratio > 0.8:
                    return f"High similarity with template (similarity: {score:.1%}{model_context}). Minor variations in wording but substantively the same."
                else:
                    return f"Matches standard template requirements (similarity: {score:.1%}{model_context}). Key terms and conditions align despite some textual differences."
            else:  # Non-Standard
                if ratio < 0.3:
                    return f"Very low similarity with template (similarity: {score:.1%}{model_context}). The contract clause differs significantly from the standard template."
                elif ratio < 0.6:
                    return f"Substantial differences from template (similarity: {score:.1%}{model_context}). Key terms or conditions deviate from the standard."
                else:
                    return f"Below threshold for standard classification (similarity: {score:.1%}{model_context}). Important variations detected in the contract language."
        
        return f"Classification: {classification} (similarity: {score:.1%}{model_context})"


def find_text_differences(contract_text, template_text):
    """Find and highlight differences between contract and template text"""
    if not contract_text or not template_text:
        return [], []
    
    contract_words = contract_text.split()
    template_words = template_text.split()
    
    matcher = difflib.SequenceMatcher(None, contract_words, template_words)
    
    matching_phrases = []
    differing_phrases = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            phrase = ' '.join(contract_words[i1:i2])
            if len(phrase) > 10:
                matching_phrases.append(phrase)
        elif tag == 'replace' or tag == 'delete':
            phrase = ' '.join(contract_words[i1:i2])
            if phrase:
                differing_phrases.append(phrase)
    
    return matching_phrases[:5], differing_phrases[:5]


def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🏥 HiLabs Healthcare Contract Classification System")
        st.markdown("### AI-Powered Contract Analysis for Healthcare Organizations")
    
    st.divider()


def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # File upload section
        st.subheader("📁 Upload Contracts")
        
        state_option = st.selectbox(
            "Select State",
            ["TN", "WA"],
            help="Choose the state for contract processing"
        )
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload contract PDF files for analysis",
            key="file_uploader"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Check if model changed and results exist
        if st.session_state.current_results:
            results_model = None
            for result in st.session_state.current_results:
                if hasattr(result, 'model_type_used'):
                    results_model = result.model_type_used
                    break
            
            if results_model and results_model != st.session_state.current_model_type:
                st.warning(f"⚠️ Current results were processed with {results_model} model. Reprocess to use {st.session_state.current_model_type} model.")
        
        if st.session_state.uploaded_files and not st.session_state.processing:
            if st.button("🚀 Process Contracts", type="primary"):
                st.session_state.processing = True
                process_uploaded_contracts(st.session_state.uploaded_files, state_option)
        elif st.session_state.processing:
            st.info("⏳ Processing in progress...")
        
        st.divider()
        
        # Processing options
        st.subheader("🎯 Processing Options")
        
        classification_model = st.selectbox(
            "Classification Model",
            ["fuzzy", "semantic", "hybrid"],
            help="Select the classification model type:\n"
                 "• Fuzzy: Fast, uses text matching (2-5 sec/contract)\n"
                 "• Semantic: Accurate, uses AI models (30-60 sec/contract)\n"
                 "• Hybrid: Best accuracy, combines both (40-80 sec/contract)",
            key="model_selector",
            index=["fuzzy", "semantic", "hybrid"].index(st.session_state.current_model_type)
        )
        
        if classification_model in ["semantic", "hybrid"]:
            st.warning("⚠️ Semantic models are slower but more accurate. First run will download models (~90MB).")
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Similarity threshold for Standard classification"
        )
        
        if st.button("Apply Settings", key="apply_settings"):
            update_model_config(classification_model, threshold)
            if st.session_state.current_results:
                st.info("💡 Model changed. Please reprocess contracts to apply new settings.")
            st.session_state.uploaded_files = []
            st.rerun()
        
        st.divider()
        
        # Export options
        st.subheader("💾 Export Options")
        
        if st.session_state.current_results:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel", "Detailed CSV"]
            )
            
            if st.button("📥 Export Results"):
                export_results(export_format)


def process_uploaded_contracts(uploaded_files, state):
    """Process uploaded contract files with optimization"""
    try:
        # Get the model type from the current config, not the selector
        config = load_configuration()
        model_type = config['classification']['model_type']
        
        # Update session state to match
        st.session_state.current_model_type = model_type
        
        if model_type == "fuzzy":
            st.info("🚀 Using Fast Fuzzy Model - Expected: 2-5 seconds per contract")
        elif model_type == "semantic":
            st.warning("🤖 Using Semantic Model - Expected: 30-60 seconds per contract")
        else:
            st.warning("⚡ Using Hybrid Model - Expected: 40-80 seconds per contract")
        
        processor = initialize_processor()
        if not processor:
            st.error("Failed to initialize processor")
            return
        
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_files = len(uploaded_files)
            
            # Get template for state (cached)
            template_path = processor.file_handler.get_template_path(state)
            if template_path:
                template_clauses = processor.process_template(template_path)
            else:
                st.error(f"No template found for state {state}")
                return
            
            for idx, uploaded_file in enumerate(uploaded_files):
                progress = (idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{total_files})")
                
                # Save uploaded file temporarily
                temp_path = Path(f"temp/{uploaded_file.name}")
                temp_path.parent.mkdir(exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    result = processor.process_contract(temp_path, template_clauses, state)
                    
                    if result:
                        # Ensure model type is set correctly
                        result.model_type_used = model_type
                        result = ensure_all_attributes(result, template_clauses)
                        results.append(result)
                        st.session_state.processed_contracts.append(result)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
            
            st.session_state.current_results = results
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully processed {len(results)} contracts!")
            
    except Exception as e:
        st.error(f"Processing error: {e}")
    finally:
        st.session_state.processing = False
    
    st.rerun()


def ensure_all_attributes(classification, template_clauses):
    """Ensure all 5 expected attributes are in the results"""
    expected_attributes = [
        "Medicaid Timely Filing",
        "Medicare Timely Filing", 
        "No Steerage/SOC",
        "Medicaid Fee Schedule",
        "Medicare Fee Schedule"
    ]
    
    present_attrs = {r.attribute_name for r in classification.results}
    
    for attr in expected_attributes:
        if attr not in present_attrs:
            missing_result = ClassificationResult(
                attribute_name=attr,
                classification="Not Found",
                similarity_score=0.0,
                reason=f"The '{attr}' clause was not found in the contract. This is a required attribute that appears to be missing.",
                contract_text="[Not Found in Contract]",
                template_text=template_clauses.get(attr, "[Not in Template]")
            )
            classification.add_result(missing_result)
    
    return classification


def display_results(results):
    """Display classification results"""
    if not results:
        st.warning("No results to display")
        return
    
    # Summary metrics
    st.header("📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_contracts = len(results)
    total_standard = sum(r.standard_count for r in results)
    total_non_standard = sum(r.non_standard_count for r in results)
    total_not_found = sum(
        sum(1 for res in r.results if res.classification == "Not Found")
        for r in results
    )
    total_attributes = sum(len(r.results) for r in results)
    
    with col1:
        st.metric("Total Contracts", total_contracts)
    with col2:
        st.metric("Standard Clauses", total_standard, delta=f"{total_standard/total_attributes*100:.1f}%")
    with col3:
        st.metric("Non-Standard Clauses", total_non_standard, delta=f"{total_non_standard/total_attributes*100:.1f}%")
    with col4:
        st.metric("Not Found", total_not_found, delta=f"{total_not_found/total_attributes*100:.1f}%")
    
    st.divider()
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Classification Details", "📈 Analytics", "🔍 Comparison View", "📄 Raw Data"])
    
    with tab1:
        display_classification_details(results)
    
    with tab2:
        display_analytics(results)
    
    with tab3:
        display_comparison_view(results)
    
    with tab4:
        display_raw_data(results)


def display_classification_details(results):
    """Display detailed classification results with reasoning"""
    st.subheader("Classification Results by Contract")
    
    expected_attributes = [
        "Medicaid Timely Filing",
        "Medicare Timely Filing",
        "No Steerage/SOC",
        "Medicaid Fee Schedule",
        "Medicare Fee Schedule"
    ]
    
    for contract in results:
        with st.expander(f"📄 {contract.contract_id} ({contract.state})", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**State:** {contract.state}")
                st.write(f"**Standard:** {contract.standard_count}")
            with col2:
                st.write(f"**Contract ID:** {contract.contract_id}")
                not_found_count = sum(1 for r in contract.results if r.classification == "Not Found")
                st.write(f"**Non-Standard:** {contract.non_standard_count}")
                st.write(f"**Not Found:** {not_found_count}")
            
            st.write("**Attribute Classifications:**")
            
            df_data = []
            results_map = {r.attribute_name: r for r in contract.results}
            
            for attr_name in expected_attributes:
                if attr_name in results_map:
                    result = results_map[attr_name]
                    
                    decision_factors = []
                    if hasattr(result, 'detailed_scores') and result.detailed_scores:
                        for score_type, score_val in result.detailed_scores.items():
                            if score_type != 'model_type' and isinstance(score_val, (int, float)) and score_val > 0:
                                decision_factors.append(f"{score_type}: {score_val:.1%}")
                    
                    df_data.append({
                        'Attribute': result.attribute_name,
                        'Classification': result.classification,
                        'Confidence': f"{result.similarity_score:.2%}" if result.classification != "Not Found" else "N/A",
                        'Reason': result.reason,
                        'Decision Factors': ', '.join(decision_factors) if decision_factors else 'Text matching'
                    })
                else:
                    df_data.append({
                        'Attribute': attr_name,
                        'Classification': 'Not Found',
                        'Confidence': 'N/A',
                        'Reason': 'Attribute not extracted from contract',
                        'Decision Factors': 'N/A'
                    })
            
            df = pd.DataFrame(df_data)
            
            def color_classification(val):
                if val == 'Standard':
                    return 'color: green; font-weight: bold'
                elif val == 'Non-Standard':
                    return 'color: red; font-weight: bold'
                elif val == 'Not Found':
                    return 'color: orange; font-weight: bold'
                return ''
            
            styled_df = df.style.map(color_classification, subset=['Classification'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
                    "Classification": st.column_config.TextColumn("Classification", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Reason": st.column_config.TextColumn("Detailed Reasoning", width="large"),
                    "Decision Factors": st.column_config.TextColumn("Score Components", width="medium")
                }
            )


@st.cache_data
def prepare_analytics_data(results):
    """Cache analytics data preparation"""
    expected_attributes = [
        "Medicaid Timely Filing",
        "Medicare Timely Filing",
        "No Steerage/SOC",
        "Medicaid Fee Schedule", 
        "Medicare Fee Schedule"
    ]
    
    all_classifications = []
    for contract in results:
        found_attrs = set()
        
        for result in contract.results:
            all_classifications.append({
                'Contract': contract.contract_id,
                'State': contract.state,
                'Attribute': result.attribute_name,
                'Classification': result.classification,
                'Confidence': result.similarity_score if result.classification != "Not Found" else 0
            })
            found_attrs.add(result.attribute_name)
        
        for attr in expected_attributes:
            if attr not in found_attrs:
                all_classifications.append({
                    'Contract': contract.contract_id,
                    'State': contract.state,
                    'Attribute': attr,
                    'Classification': 'Not Found',
                    'Confidence': 0
                })
    
    return pd.DataFrame(all_classifications)


def display_analytics(results):
    """Display analytics and visualizations"""
    st.subheader("Classification Analytics")
    
    df = prepare_analytics_data(results)
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Classification distribution pie chart
            fig_pie = px.pie(
                df,
                names='Classification',
                title='Overall Classification Distribution',
                color_discrete_map={
                    'Standard': '#28a745',
                    'Non-Standard': '#dc3545',
                    'Not Found': '#ffc107'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution histogram
            df_with_confidence = df[df['Classification'] != 'Not Found']
            if not df_with_confidence.empty:
                fig_hist = px.histogram(
                    df_with_confidence,
                    x='Confidence',
                    nbins=20,
                    title='Confidence Score Distribution',
                    labels={'Confidence': 'Confidence Score', 'count': 'Frequency'}
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No confidence scores available")
        
        # Classification by attribute
        expected_attributes = [
            "Medicaid Timely Filing",
            "Medicare Timely Filing",
            "No Steerage/SOC",
            "Medicaid Fee Schedule", 
            "Medicare Fee Schedule"
        ]
        
        attribute_counts = []
        for attr in expected_attributes:
            attr_df = df[df['Attribute'] == attr]
            for classification in ['Standard', 'Non-Standard', 'Not Found']:
                count = len(attr_df[attr_df['Classification'] == classification])
                attribute_counts.append({
                    'Attribute': attr,
                    'Classification': classification,
                    'Count': count
                })
        
        attr_df = pd.DataFrame(attribute_counts)
        
        fig_bar = px.bar(
            attr_df,
            x='Attribute',
            y='Count',
            color='Classification',
            title='Classifications by Attribute',
            color_discrete_map={
                'Standard': '#28a745',
                'Non-Standard': '#dc3545', 
                'Not Found': '#ffc107'
            }
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence heatmap
        pivot_df = df.pivot_table(
            values='Confidence',
            index='Contract',
            columns='Attribute',
            aggfunc='mean',
            fill_value=0
        )
        
        for attr in expected_attributes:
            if attr not in pivot_df.columns:
                pivot_df[attr] = 0
        
        pivot_df = pivot_df[expected_attributes]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            zmid=0.5,
            text=pivot_df.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(
            title='Confidence Scores Heatmap',
            xaxis_title='Attribute',
            yaxis_title='Contract',
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


def display_comparison_view(results):
    """Display enhanced side-by-side comparison with detailed analysis"""
    st.subheader("Contract vs Template Comparison")
    
    if not results:
        st.warning("No results available for comparison")
        return
    
    contract_options = [f"{r.contract_id} ({r.state})" for r in results]
    selected_contract_idx = st.selectbox(
        "Select Contract", 
        range(len(contract_options)), 
        format_func=lambda x: contract_options[x],
        key="contract_selector_comparison"
    )
    
    if selected_contract_idx is not None:
        selected_contract = results[selected_contract_idx]
        
        attribute_options = [
            "Medicaid Timely Filing",
            "Medicare Timely Filing",
            "No Steerage/SOC", 
            "Medicaid Fee Schedule",
            "Medicare Fee Schedule"
        ]
        
        selected_attribute = st.selectbox(
            "Select Attribute", 
            attribute_options,
            key="attribute_selector_comparison"
        )
        
        if selected_attribute:
            selected_result = next(
                (r for r in selected_contract.results if r.attribute_name == selected_attribute),
                None
            )
            
            if selected_result:
                st.markdown("---")
                
                classification_color = {
                    "Standard": "🟢",
                    "Non-Standard": "🔴",
                    "Not Found": "🟡"
                }.get(selected_result.classification, "⚪")
                
                st.markdown(f"### {classification_color} Classification: **{selected_result.classification}**")
                
                if selected_result.classification != "Not Found":
                    st.markdown(f"**Confidence Score:** {selected_result.similarity_score:.1%}")
                
                with st.expander("📊 **Detailed Analysis & Reasoning**", expanded=True):
                    st.markdown("#### Classification Decision")
                    st.info(selected_result.reason)
                    
                    if selected_result.classification == "Non-Standard":
                        contract_text = selected_result.contract_text or ""
                        template_text = selected_result.template_text or ""
                        
                        if contract_text and template_text:
                            matching_phrases, differing_phrases = find_text_differences(
                                contract_text, template_text
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("##### 🔍 Key Differences Found:")
                                if differing_phrases:
                                    for phrase in differing_phrases:
                                        st.markdown(f"- *{phrase}*")
                                else:
                                    st.markdown("- Structure or context differs")
                            
                            with col2:
                                st.markdown("##### ✅ Matching Elements:")
                                if matching_phrases:
                                    for phrase in matching_phrases[:3]:
                                        if len(phrase) > 50:
                                            phrase = phrase[:50] + "..."
                                        st.markdown(f"- {phrase}")
                                else:
                                    st.markdown("- Limited textual overlap")
                    
                    # Show score breakdown if available
                    if hasattr(selected_result, 'detailed_scores') and selected_result.detailed_scores:
                        st.markdown("#### Score Components")
                        
                        result_model_type = selected_result.detailed_scores.get('model_type', 'fuzzy')
                        st.caption(f"*Processed with {result_model_type} model*")
                        
                        score_items = []
                        
                        if 'fuzzy_score' in selected_result.detailed_scores:
                            fuzzy_val = selected_result.detailed_scores['fuzzy_score']
                            if isinstance(fuzzy_val, (int, float)):
                                score_items.append(('Fuzzy Score', f"{fuzzy_val:.1%}"))
                        
                        if 'exact_score' in selected_result.detailed_scores:
                            exact_val = selected_result.detailed_scores['exact_score']
                            if isinstance(exact_val, (int, float)):
                                score_items.append(('Exact Match Score', f"{exact_val:.1%}"))
                        
                        if 'semantic_score' in selected_result.detailed_scores:
                            semantic_val = selected_result.detailed_scores['semantic_score']
                            if result_model_type == 'fuzzy':
                                score_items.append(('Semantic Score', 'N/A (Fuzzy Mode)'))
                            elif isinstance(semantic_val, (int, float)):
                                if semantic_val > 0:
                                    score_items.append(('Semantic Score', f"{semantic_val:.1%}"))
                                else:
                                    score_items.append(('Semantic Score', '0.0%'))
                        
                        for score_type, score_val in selected_result.detailed_scores.items():
                            if score_type not in ['fuzzy_score', 'exact_score', 'semantic_score', 'model_type']:
                                if isinstance(score_val, (int, float)) and score_val > 0:
                                    name = score_type.replace('_score', '').replace('_', ' ').title()
                                    score_items.append((name, f"{score_val:.1%}"))
                        
                        if score_items:
                            score_cols = st.columns(len(score_items))
                            for idx, (name, value) in enumerate(score_items):
                                with score_cols[idx]:
                                    st.metric(name, value)
                        
                        current_model = st.session_state.get('current_model_type', 'fuzzy')
                        if result_model_type != current_model:
                            st.warning(f"⚠️ These results were generated using {result_model_type} model. Current model is {current_model}. Reprocess to update scores.")
                
                st.markdown("---")
                st.markdown("### 📝 Text Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Contract Text")
                    contract_text = selected_result.contract_text if selected_result.contract_text else "[Not Found in Contract]"
                    
                    if selected_result.classification == "Non-Standard" and contract_text != "[Not Found in Contract]":
                        st.markdown("*Highlighted areas show deviations from template*")
                    
                    st.text_area(
                        "Contract",
                        contract_text,
                        height=400,
                        disabled=True,
                        key=f"contract_text_{selected_contract_idx}_{selected_attribute}",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    st.markdown("##### Template Text")
                    template_text = selected_result.template_text if selected_result.template_text else "[Not in Template]"
                    
                    st.text_area(
                        "Template",
                        template_text,
                        height=400,
                        disabled=True,
                        key=f"template_text_{selected_contract_idx}_{selected_attribute}",
                        label_visibility="collapsed"
                    )
                
                if selected_result.classification == "Not Found":
                    st.warning("⚠️ This attribute was expected but not found in the contract.")
                elif selected_result.classification == "Non-Standard":
                    st.warning("⚠️ This clause deviates from the standard template.")
                elif selected_result.classification == "Standard":
                    st.success("✅ This clause aligns with the standard template requirements.")
            else:
                st.warning(f"Attribute '{selected_attribute}' not found in results")


def display_raw_data(results):
    """Display raw data with enhanced detail"""
    st.subheader("Raw Classification Data")
    
    data = []
    for contract in results:
        for result in contract.results:
            matching_info = ""
            if hasattr(result, 'matching_phrases') and result.matching_phrases:
                matching_info = f"Matches: {', '.join(result.matching_phrases[:3])}"
            
            score_breakdown = ""
            if hasattr(result, 'detailed_scores') and result.detailed_scores:
                scores = []
                for score_type, score_val in result.detailed_scores.items():
                    if score_type != 'model_type' and isinstance(score_val, (int, float)) and score_val > 0:
                        scores.append(f"{score_type}: {score_val:.2f}")
                score_breakdown = '; '.join(scores)
            
            contract_snippet = ""
            if result.contract_text and result.contract_text != "[Not Found in Contract]":
                contract_snippet = result.contract_text[:100] + "..." if len(result.contract_text) > 100 else result.contract_text
            
            data.append({
                'Contract ID': contract.contract_id,
                'State': contract.state,
                'Attribute': result.attribute_name,
                'Classification': result.classification,
                'Confidence': result.similarity_score,
                'Detailed Reason': result.reason,
                'Score Components': score_breakdown or 'N/A',
                'Contract Text Preview': contract_snippet or '[Not Found]',
                'Matching Phrases': matching_info or 'N/A',
                'Timestamp': result.timestamp if hasattr(result, 'timestamp') else datetime.now()
            })
    
    df = pd.DataFrame(data)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Similarity score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Detailed Reason": st.column_config.TextColumn(
                "Detailed Reason",
                width="large"
            ),
            "Contract Text Preview": st.column_config.TextColumn(
                "Text Preview",
                width="medium"
            )
        }
    )
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Detailed CSV",
        data=csv,
        file_name=f"detailed_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_csv_detailed_{datetime.now().timestamp()}"
    )


def export_results(format_type):
    """Export results in specified format with full details"""
    if not st.session_state.current_results:
        st.error("No results to export")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format_type == "CSV":
        data = []
        for contract in st.session_state.current_results:
            for result in contract.results:
                data.append({
                    'Contract ID': contract.contract_id,
                    'State': contract.state,
                    'Attribute': result.attribute_name,
                    'Classification': result.classification,
                    'Confidence': result.similarity_score,
                    'Reason': result.reason
                })
        
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"contract_classification_{timestamp}.csv",
            mime="text/csv",
            key=f"export_csv_{timestamp}"
        )
        
    elif format_type == "Detailed CSV":
        data = []
        for contract in st.session_state.current_results:
            for result in contract.results:
                row = {
                    'Contract ID': contract.contract_id,
                    'State': contract.state,
                    'Attribute': result.attribute_name,
                    'Classification': result.classification,
                    'Confidence': result.similarity_score,
                    'Detailed Reason': result.reason,
                    'Contract Text': result.contract_text if result.contract_text else '[Not Found]',
                    'Template Text': result.template_text if result.template_text else '[Not in Template]',
                    'Timestamp': result.timestamp if hasattr(result, 'timestamp') else datetime.now()
                }
                
                if hasattr(result, 'detailed_scores') and result.detailed_scores:
                    for score_type, score_val in result.detailed_scores.items():
                        if score_type != 'model_type':
                            row[f'Score_{score_type}'] = score_val
                
                data.append(row)
        
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Detailed CSV",
            data=csv,
            file_name=f"detailed_contract_classification_{timestamp}.csv",
            mime="text/csv",
            key=f"export_detailed_csv_{timestamp}"
        )
        
    elif format_type == "JSON":
        json_data = []
        for contract in st.session_state.current_results:
            contract_dict = {
                'contract_id': contract.contract_id,
                'state': contract.state,
                'standard_count': contract.standard_count,
                'non_standard_count': contract.non_standard_count,
                'results': []
            }
            
            for result in contract.results:
                result_dict = {
                    'attribute_name': result.attribute_name,
                    'classification': result.classification,
                    'similarity_score': result.similarity_score,
                    'reason': result.reason,
                    'contract_text': result.contract_text,
                    'template_text': result.template_text,
                    'timestamp': str(result.timestamp if hasattr(result, 'timestamp') else datetime.now())
                }
                
                if hasattr(result, 'detailed_scores'):
                    result_dict['detailed_scores'] = result.detailed_scores
                
                contract_dict['results'].append(result_dict)
            
            json_data.append(contract_dict)
        
        json_str = json.dumps(json_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"contract_classification_{timestamp}.json",
            mime="application/json",
            key=f"export_json_{timestamp}"
        )
        
    elif format_type == "Excel":
        import io
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            summary_data = []
            for contract in st.session_state.current_results:
                not_found_count = sum(1 for r in contract.results if r.classification == "Not Found")
                summary_data.append({
                    'Contract ID': contract.contract_id,
                    'State': contract.state,
                    'Standard Count': contract.standard_count,
                    'Non-Standard Count': contract.non_standard_count,
                    'Not Found Count': not_found_count,
                    'Total Attributes': len(contract.results)
                })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            detailed_data = []
            for contract in st.session_state.current_results:
                for result in contract.results:
                    detailed_data.append({
                        'Contract ID': contract.contract_id,
                        'State': contract.state,
                        'Attribute': result.attribute_name,
                        'Classification': result.classification,
                        'Confidence': result.similarity_score,
                        'Detailed Reason': result.reason,
                        'Contract Text (First 500 chars)': result.contract_text[:500] if result.contract_text else '[Not Found]',
                        'Template Text (First 500 chars)': result.template_text[:500] if result.template_text else '[Not in Template]'
                    })
            
            pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed Results', index=False)
        
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"contract_classification_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_excel_{timestamp}"
        )


def main():
    """Main application entry point"""
    render_header()
    render_sidebar()
    
    if st.session_state.current_results:
        display_results(st.session_state.current_results)
    else:
        st.markdown("""
        ## Welcome to the HiLabs Contract Classification System
        
        This AI-powered system helps healthcare organizations automatically:
        - 📄 **Extract** key clauses from healthcare contracts
        - 🔍 **Compare** extracted clauses with standard templates
        - 🎯 **Classify** each clause as Standard or Non-Standard
        - 📊 **Generate** detailed analytics and reports
        
        ### Getting Started
        1. Select a state (TN or WA) in the sidebar
        2. Upload one or more contract PDF files
        3. Click "Process Contracts" to begin analysis
        4. Review results and export as needed
        
        ### Key Attributes Analyzed
        - **Medicaid Timely Filing** - Claims submission deadlines
        - **Medicare Timely Filing** - Medicare claims deadlines
        - **No Steerage/SOC** - Network participation rules
        - **Medicaid Fee Schedule** - Payment methodology
        - **Medicare Fee Schedule** - Medicare payment rates
        
        ### Classification Models
        - **Fuzzy** (Fast): Text-based matching, 2-5 seconds per contract
        - **Semantic** (Accurate): AI-powered understanding, 30-60 seconds per contract
        - **Hybrid** (Best): Combines both approaches, 40-80 seconds per contract
        """)
        
        st.info(f"🔧 Current Model: {st.session_state.current_model_type.upper()}")


if __name__ == "__main__":
    main()