import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import traceback
import signal

# Add the project root to PYTHONPATH
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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

def signal_handler(signum, frame):
    logger.error(f"Received signal {signum}. Exiting...")
    sys.exit(1)

signal.signal(signal.SIGSEGV, signal_handler)

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize models
        initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("Application will start, but some features may not work correctly.")

if __name__ == "__main__":
    try:
        logger.info(f"Starting server on {config['server']['host']}:{config['server']['port']}")
        uvicorn.run(app, host=config['server']['host'], port=int(config['server']['port']))
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

