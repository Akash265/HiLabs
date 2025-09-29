"""
Text comparison and similarity calculation module
"""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import Logger
from src.utils.file_handler import CacheManager

logger = Logger.setup_logger(__name__)

# Try to import sentence-transformers, fall back to simple version if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence transformers available")
except ImportError as e:
    logger.warning(f"Sentence transformers not available: {e}")
    logger.info("Using fuzzy matching only")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class SimilarityScores:
    """Container for similarity scores"""
    exact_match: float
    fuzzy_score: float
    semantic_score: float
    weighted_score: float
    
    def to_dict(self) -> Dict:
        return {
            'exact_match': self.exact_match,
            'fuzzy_score': self.fuzzy_score,
            'semantic_score': self.semantic_score,
            'weighted_score': self.weighted_score
        }


class TextComparator:
    """Compare text using multiple similarity metrics"""
    
    def __init__(self, config: Dict):
        """
        Initialize text comparator
        
        Args:
            config: Configuration with weights and thresholds
        """
        self.config = config
        
        # Initialize semantic model if available and enabled
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and config.get('semantic', {}).get('enabled', True):
            model_name = config['semantic'].get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            try:
                logger.info(f"Loading semantic model: {model_name}")
                self.semantic_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
                self.semantic_model = None
        
        # Initialize cache for embeddings
        self.cache_manager = CacheManager()
        
        # Get weights
        if self.semantic_model:
            self.weights = {
                'exact': config.get('exact', {}).get('weight', 0.2),
                'fuzzy': config.get('fuzzy', {}).get('weight', 0.3),
                'semantic': config.get('semantic', {}).get('weight', 0.5)
            }
        else:
            # Adjust weights if semantic model not available
            self.weights = {
                'exact': 0.3,
                'fuzzy': 0.7,
                'semantic': 0.0
            }
            logger.info("Using fuzzy matching only (semantic model not available)")
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate weighted similarity score
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Weighted similarity score (0-1)
        """
        scores = self.get_detailed_scores(text1, text2)
        return scores.weighted_score
        
    def get_detailed_scores(self, text1: str, text2: str) -> SimilarityScores:
        """
        Get all similarity scores
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Detailed similarity scores
        """
        # Normalize texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        # Calculate individual scores
        exact_score = self.exact_match(norm_text1, norm_text2)
        fuzzy_score = self.fuzzy_similarity(norm_text1, norm_text2)
        
        # Only calculate semantic if model is available and enabled
        if self.semantic_model and self.weights.get('semantic', 0) > 0:
            semantic_score = self.semantic_similarity(text1, text2)
        else:
            semantic_score = 0.0
        
        # Calculate weighted score based on active components
        if self.semantic_model and self.weights.get('semantic', 0) > 0:
            # All three components active
            weighted_score = (
                self.weights['exact'] * exact_score +
                self.weights['fuzzy'] * fuzzy_score +
                self.weights['semantic'] * semantic_score
            )
        else:
            # Only exact and fuzzy active - recalculate weights
            total_weight = self.weights['exact'] + self.weights['fuzzy']
            if total_weight > 0:
                weighted_score = (
                    (self.weights['exact'] / total_weight) * exact_score +
                    (self.weights['fuzzy'] / total_weight) * fuzzy_score
                )
            else:
                weighted_score = fuzzy_score  # Fallback to fuzzy score
        
        return SimilarityScores(
            exact_match=exact_score,
            fuzzy_score=fuzzy_score,
            semantic_score=semantic_score,
            weighted_score=weighted_score
        )
        
    def exact_match(self, text1: str, text2: str) -> float:
        """
        Check for exact match
        
        Args:
            text1: First text (normalized)
            text2: Second text (normalized)
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if text1 == text2 else 0.0
        
    def fuzzy_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate fuzzy string similarity
        
        Args:
            text1: First text (normalized)
            text2: Second text (normalized)
            
        Returns:
            Fuzzy similarity score (0-1)
        """
        # Use multiple fuzzy matching strategies
        scores = []
        
        # Token set ratio - handles different word order
        token_set = fuzz.token_set_ratio(text1, text2) / 100.0
        scores.append(token_set)
        
        # Partial ratio - handles substring matching
        partial = fuzz.partial_ratio(text1, text2) / 100.0
        scores.append(partial * 0.8)  # Weight partial matches lower
        
        # Token sort ratio - handles sorted tokens
        token_sort = fuzz.token_sort_ratio(text1, text2) / 100.0
        scores.append(token_sort)
        
        # Return weighted average
        return np.mean(scores)
        
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using sentence transformers
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score (0-1)
        """
        if not self.semantic_model:
            return 0.0
            
        try:
            # Get embeddings (with caching)
            embedding1 = self._get_embedding(text1)
            embedding2 = self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            # Ensure in [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
            
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding with caching
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Check cache
        cache_key = self.cache_manager.get_cache_key(text, prefix="emb")
        cached_embedding = self.cache_manager.load(cache_key)
        
        if cached_embedding is not None:
            return np.array(cached_embedding)
            
        # Generate embedding
        embedding = self.semantic_model.encode(text, show_progress_bar=False)
        
        # Cache for future use
        self.cache_manager.save(cache_key, embedding.tolist())
        
        return embedding
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except periods and commas
        text = re.sub(r'[^\w\s.,%-]', '', text)
        
        # Normalize percentage formats
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        
        # Normalize common abbreviations
        replacements = {
            'medicare advantage': 'medicare',
            'days from': 'days of',
            'calendar days': 'days',
            'business days': 'days'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text.strip()
        
    def handle_placeholders(self, text: str) -> str:
        """
        Handle placeholder values in templates
        
        Args:
            text: Text with potential placeholders
            
        Returns:
            Text with normalized placeholders
        """
        # Replace common placeholders
        text = re.sub(r'\[?XX+%?\]?', '[PERCENTAGE]', text)
        text = re.sub(r'\[Fee Schedule\]', 'Fee Schedule', text)
        text = re.sub(r'\[.*?\]', '[PLACEHOLDER]', text)
        
        return text
        
    def apply_business_rules(self, contract_text: str, template_text: str) -> str:
        """
        Apply business rules to determine classification
        
        Args:
            contract_text: Extracted contract clause
            template_text: Standard template clause
            
        Returns:
            Reason for classification
        """
        contract_norm = self.normalize_text(contract_text)
        template_norm = self.normalize_text(template_text)
        
        # Check for structural changes
        if 'except' in contract_norm and 'except' not in template_norm:
            return "Contains exception not in template"
            
        if 'carve out' in contract_norm or 'carveout' in contract_norm:
            return "Contains carve-out provision"
            
        # Check for value substitution
        template_values = re.findall(r'(\d+)%?', template_norm)
        contract_values = re.findall(r'(\d+)%?', contract_norm)
        
        if template_values and contract_values:
            # Check if values are within acceptable range
            for tv, cv in zip(template_values, contract_values):
                if abs(int(tv) - int(cv)) > 10:  # More than 10% difference
                    return f"Significant value difference: {cv} vs {tv}"
                    
        # Check for additional conditions
        contract_conditions = len(re.findall(r'\bif\b|\bwhen\b|\bunless\b', contract_norm))
        template_conditions = len(re.findall(r'\bif\b|\bwhen\b|\bunless\b', template_norm))
        
        if contract_conditions > template_conditions:
            return "Contains additional conditions"
            
        return ""


class PlaceholderHandler:
    """Handle placeholder replacement and matching"""
    
    @staticmethod
    def extract_values(text: str) -> Dict[str, str]:
        """
        Extract actual values from text
        
        Args:
            text: Text containing values
            
        Returns:
            Dictionary of placeholder to actual value mappings
        """
        values = {}
        
        # Extract percentages
        percent_matches = re.findall(r'(\d+)\s*%', text)
        if percent_matches:
            values['[PERCENTAGE]'] = f"{percent_matches[0]}%"
            
        # Extract day periods
        days_matches = re.findall(r'(\d+)\s*days?', text, re.IGNORECASE)
        if days_matches:
            values['[DAYS]'] = f"{days_matches[0]} days"
            
        # Extract fee schedule references
        fee_match = re.search(r'(Medicare|Medicaid|[\w\s]+)\s+Fee Schedule', text, re.IGNORECASE)
        if fee_match:
            values['[FEE_SCHEDULE]'] = fee_match.group(0)
            
        return values
        
    @staticmethod
    def replace_placeholders(template: str, values: Dict[str, str]) -> str:
        """
        Replace placeholders in template with actual values
        
        Args:
            template: Template text with placeholders
            values: Dictionary of placeholder to value mappings
            
        Returns:
            Template with replaced values
        """
        result = template
        
        for placeholder, value in values.items():
            result = result.replace(placeholder, value)
            
        return result