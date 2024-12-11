import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import traceback
import signal
import firebase_admin
from firebase_admin import credentials, firestore

# Setup project root and update PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load configuration
config_path = os.path.join(project_root, "config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

from app.api.routes import router
from app.utils.logger import setup_logger
from app.services.ocr_service import initialize_models

# Setup logger
logger = setup_logger(config['logging']['level'], config['logging']['file'])


# Initialize Firebase
cred_path = os.path.join(project_root, config['firebase']['credentials_path'])
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['server']['allowed_hosts'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

# Define signal handler for graceful shutdown
def signal_handler(signum, frame):
    logger.error(f"Received signal {signum}. Exiting...")
    sys.exit(1)

# Register signal handlers for safe shutdown
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)

@app.on_event("startup")
async def startup_event():
    try:
        initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error("Error initializing models", exc_info=True)
        traceback.print_exc()
        sys.exit(1)  # Exit if models cannot be initialized


if __name__ == "__main__":
    try:
        logger.info(f"Starting server on {config['server']['host']}:{config['server']['port']}")
        uvicorn.run(app, host=config['server']['host'], port=int(config['server']['port']))
    except Exception as e:
        logger.error("Server failed to start", exc_info=True)
        sys.exit(1)
