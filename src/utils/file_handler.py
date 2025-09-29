"""
File handling utilities for contract and template management
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import json
from datetime import datetime

from src.utils.logger import Logger

logger = Logger.setup_logger(__name__)


class FileHandler:
    """Manages file I/O operations for contracts and templates"""
    
    def __init__(self, contracts_dir: str, templates_dir: str):
        """
        Initialize file handler
        
        Args:
            contracts_dir: Directory containing contract PDFs
            templates_dir: Directory containing standard template PDFs
        """
        self.contracts_dir = Path(contracts_dir)
        self.templates_dir = Path(templates_dir)
        self._validate_structure()
        
    def _validate_structure(self):
        """Validate expected directory structure"""
        if not self.contracts_dir.exists():
            raise FileNotFoundError(f"Contracts directory not found: {self.contracts_dir}")
            
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
            
        # Check for state subdirectories
        expected_states = ['TN', 'WA']
        for state in expected_states:
            state_dir = self.contracts_dir / state
            if not state_dir.exists():
                logger.warning(f"State directory not found: {state_dir}")
                
    def list_files(self, directory: Path, pattern: str = "*.pdf") -> List[Path]:
        """
        List files in directory matching pattern
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
        return sorted(files)
        
    def get_state_contracts(self, state: str) -> List[Path]:
        """
        Get all contract files for a specific state
        
        Args:
            state: State code (e.g., 'TN', 'WA')
            
        Returns:
            List of contract file paths
        """
        state_dir = self.contracts_dir / state
        if not state_dir.exists():
            logger.warning(f"State directory not found: {state_dir}")
            return []
            
        contracts = self.list_files(state_dir, "*.pdf")
        logger.info(f"Found {len(contracts)} contracts for state {state}")
        return contracts
        
    def get_template_path(self, state: str) -> Optional[Path]:
        """
        Get template file path for a state
        
        Args:
            state: State code
            
        Returns:
            Template file path or None if not found
        """
        template_path = self.templates_dir / f"{state}.pdf"
        if not template_path.exists():
            # Try alternative naming patterns
            template_path = self.templates_dir / f"{state.lower()}.pdf"
            if not template_path.exists():
                logger.error(f"Template not found for state {state}")
                return None
                
        logger.info(f"Found template for state {state}: {template_path}")
        return template_path
        
    def read_file(self, file_path: Path) -> bytes:
        """
        Read file content
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as bytes
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
            
    def get_file_info(self, file_path: Path) -> Dict:
        """
        Get file metadata
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file metadata
        """
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hash': self._calculate_file_hash(file_path)
        }
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def validate_pdf(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate if file is a valid PDF
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    return False, "File is not a valid PDF"
                    
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 50:  # Max 50MB
                return False, f"File too large: {size_mb:.1f}MB (max 50MB)"
                
            return True, ""
            
        except Exception as e:
            return False, f"Error validating PDF: {str(e)}"


class OutputManager:
    """Manages output file creation and organization"""
    
    def __init__(self, output_dir: str):
        """
        Initialize output manager
        
        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.session_dir = self._create_session_directory()
        
    def _create_session_directory(self) -> Path:
        """Create timestamped session directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (session_dir / "reports").mkdir(exist_ok=True)
        (session_dir / "exports").mkdir(exist_ok=True)
        (session_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        return session_dir
        
    def get_output_path(self, filename: str, subdir: str = "exports") -> Path:
        """
        Get output file path
        
        Args:
            filename: Output filename
            subdir: Subdirectory within session directory
            
        Returns:
            Full output path
        """
        output_path = self.session_dir / subdir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
        
    def save_json(self, data: Dict, filename: str) -> Path:
        """
        Save data as JSON
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.get_output_path(filename, "exports")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {output_path}")
        return output_path
        
    def save_text(self, content: str, filename: str) -> Path:
        """
        Save text content to file
        
        Args:
            content: Text content
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.get_output_path(filename, "reports")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved text to {output_path}")
        return output_path


class CacheManager:
    """Manages caching for expensive operations"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, data: str, prefix: str = "") -> str:
        """Generate cache key from data"""
        hash_obj = hashlib.sha256(data.encode())
        key = hash_obj.hexdigest()[:16]
        return f"{prefix}_{key}" if prefix else key
        
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file"""
        return self.cache_dir / f"{cache_key}.cache"
        
    def exists(self, cache_key: str) -> bool:
        """Check if cache exists"""
        return self.get_cache_path(cache_key).exists()
        
    def load(self, cache_key: str) -> Optional[any]:
        """Load data from cache"""
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache {cache_key}: {e}")
            return None
            
    def save(self, cache_key: str, data: any):
        """Save data to cache"""
        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str)
            logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cache {cache_key}: {e}")
            
    def clear(self):
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        logger.info("Cache cleared")