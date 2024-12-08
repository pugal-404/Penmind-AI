import logging
import os
from logging.handlers import RotatingFileHandler
import yaml

def setup_logger(log_level, log_file):
    # Load configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    with open(os.path.join(project_root, "config", "config.yaml"), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(project_root, config['logging']['directory'])
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=config['logging']['max_size'],
        backupCount=config['logging']['backup_count']
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def get_logger(name):
    return logging.getLogger(name)

