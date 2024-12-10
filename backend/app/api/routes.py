from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.security import OAuth2PasswordBearer
from app.services.ocr_service import recognize_text, retrain_model
from app.utils.logger import get_logger
import base64
import io
from PIL import Image
import traceback
import time
from firebase_admin import firestore

router = APIRouter()
logger = get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/recognize")
async def recognize_handwriting(file: UploadFile = File(...), model_type: str = 'ensemble'):
    try:
        start_time = time.time()
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        recognized_text, confidence = recognize_text(contents, model_type)
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully recognized text from uploaded image using {model_type} model")
        return {
            "text": recognized_text,
            "confidence": confidence,
            "processing_time": processing_time,
            "model_used": model_type
        }
    except Exception as e:
        logger.error(f"Error during text recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_batch")
async def recognize_batch(files: list[UploadFile] = File(...), model_type: str = 'ensemble'):
    try:
        results = []
        for file in files:
            contents = await file.read()
            if contents:
                recognized_text, confidence = recognize_text(contents, model_type)
                results.append({
                    "filename": file.filename,
                    "text": recognized_text,
                    "confidence": confidence
                })
        
        logger.info(f"Successfully processed batch of {len(files)} images")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error during batch recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def handle_feedback(feedback: dict, token: str = Depends(oauth2_scheme)):
    try:
        # Verify token (implement your own auth logic)
        if not verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Store feedback in Firebase
        db = firestore.client()
        db.collection('feedback').add(feedback)
        
        # Trigger model retraining
        retrain_model(feedback)
        
        logger.info("Feedback stored and retraining initiated")
        return {"message": "Feedback processed successfully"}
    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def verify_token(token: str):
    # Implement your token verification logic here
    # This is a placeholder implementation
    return True

