import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime


def setup_logging(app):
    """Configure logging for the application"""
    log_dir = os.path.join(app.config['STORAGE_PATH'], 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters with file and line number
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(app.config.get('LOG_LEVEL', 'INFO'))

    # Remove any existing handlers
    root_logger.handlers = []

    # Add our configured handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress certain verbose libraries if needed
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # Log startup message
    root_logger.info(f"Starting application with data directory: {app.config['STORAGE_PATH']}")
