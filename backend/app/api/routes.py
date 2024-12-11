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
async def recognize_handwriting(file: UploadFile = File(...), model_type: str = Form('ensemble')):
    try:
        start_time = time.time()
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        image = Image.open(io.BytesIO(contents))
        recognized_text, confidence = recognize_text(image, model_type)
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully recognized text from uploaded image using {model_type} model.")
        return {
            "text": recognized_text,
            "confidence": confidence,
            "processing_time": processing_time,
            "model_used": model_type
        }
    except Exception as e:
        logger.error("Error during text recognition", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/recognize_batch")
async def recognize_batch(files: list[UploadFile] = File(...), model_type: str = Form('ensemble')):
    results = []
    for file in files:
        try:
            contents = await file.read()
            if contents:
                image = Image.open(io.BytesIO(contents))
                recognized_text, confidence = recognize_text(image, model_type)
                results.append({
                    "filename": file.filename,
                    "text": recognized_text,
                    "confidence": confidence
                })
        except Exception as e:
            logger.error(f"Failed processing {file.filename}: {str(e)}", exc_info=True)
            results.append({
                "filename": file.filename,
                "error": "Failed to process image"
            })
        
    logger.info(f"Successfully processed batch of {len(files)} images.")
    return {"results": results}

@router.post("/feedback")
async def handle_feedback(feedback: dict, token: str = Depends(oauth2_scheme)):
    try:
        if not verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        db = firestore.client()
        db.collection('feedback').add(feedback)
        retrain_model(feedback)
        
        logger.info("Feedback stored and retraining initiated.")
        return {"message": "Feedback processed successfully"}
    except Exception as e:
        logger.error("Error handling feedback", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

def verify_token(token: str):
    # Placeholder implementation
    return True