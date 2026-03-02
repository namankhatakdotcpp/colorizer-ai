import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for production-ready structured logging.
    Captures standard log fields and any extra kwargs dynamically.
    """
    def format(self, record):
        # Base attributes
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Standard attributes to ignore (so we only capture custom extras)
        standard_attrs = {'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName', 'taskName'}
        
        # Add extra custom attributes
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

def setup_logging():
    """
    Configures the root logger to output JSON to both stdout and a rolling file.
    Must be called at application startup.
    """
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing default handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    formatter = JSONFormatter()
        
    # JSON Console Handler (For Docker stdout / Datadog / ELK ingestion)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # JSON Rotating File Handler (For local disk persistence)
    log_file = os.path.join(log_dir, "colorizer.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024, # 20MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Silence third-party noise
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    
    logger.info("Structured JSON logging initialized.")
