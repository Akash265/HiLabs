"""
Logging configuration for the HiLabs Contract Classification System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog


class Logger:
    """Centralized logging configuration with color output and file logging"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def setup_logger(
        name: str,
        log_file: str = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Set up a logger with color output and optional file logging
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Configured logger instance
        """
        
        # Return existing logger if already configured
        if name in Logger._loggers:
            return Logger._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplication
        logger.handlers = []
        
        # Console handler with color
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Color formatter for console
        console_format = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler if log file specified
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        # Store logger reference
        Logger._loggers[name] = logger
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get existing logger or create new one"""
        if name in Logger._loggers:
            return Logger._loggers[name]
        return Logger.setup_logger(name)


class ProcessingLogger:
    """Specialized logger for tracking document processing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
        
    def start_processing(self, doc_id: str, doc_name: str):
        """Log start of document processing"""
        self.start_times[doc_id] = datetime.now()
        self.logger.info(f"Starting processing: {doc_name} (ID: {doc_id})")
        
    def end_processing(self, doc_id: str, doc_name: str, success: bool = True):
        """Log end of document processing with duration"""
        if doc_id in self.start_times:
            duration = (datetime.now() - self.start_times[doc_id]).total_seconds()
            status = "completed successfully" if success else "failed"
            self.logger.info(
                f"Processing {status}: {doc_name} (ID: {doc_id}) - "
                f"Duration: {duration:.2f} seconds"
            )
            del self.start_times[doc_id]
            
    def log_extraction(self, doc_id: str, attribute: str, success: bool, details: str = ""):
        """Log attribute extraction results"""
        level = logging.INFO if success else logging.WARNING
        status = "extracted" if success else "failed to extract"
        message = f"Document {doc_id}: {status} '{attribute}'"
        if details:
            message += f" - {details}"
        self.logger.log(level, message)
        
    def log_classification(
        self, 
        doc_id: str, 
        attribute: str, 
        classification: str, 
        confidence: float
    ):
        """Log classification results"""
        self.logger.info(
            f"Document {doc_id}: Classified '{attribute}' as {classification} "
            f"(confidence: {confidence:.2%})"
        )
        
    def log_error(self, doc_id: str, error: Exception, context: str = ""):
        """Log processing errors"""
        message = f"Error processing document {doc_id}"
        if context:
            message += f" during {context}"
        message += f": {str(error)}"
        self.logger.error(message, exc_info=True)


# Create default logger instance
default_logger = Logger.setup_logger("contract_classification", "logs/classification.log")