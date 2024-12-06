# backend/app/utils/logger.py
import logging
import yaml
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    log_dir = config["logging"]["directory"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, config["logging"]["file"])

    handler = RotatingFileHandler(
        log_file, 
        maxBytes=config["logging"]["max_size"], 
        backupCount=config["logging"]["backup_count"]
    )
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(config["logging"]["level"])
    logger.addHandler(handler)

    # Add console handler for development environment
    if config["environment"] == "development":
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

