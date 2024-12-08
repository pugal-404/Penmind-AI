from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from app.services.ocr_service import recognize_text
from app.utils.logger import get_logger
import base64
import io
from PIL import Image
import traceback

router = APIRouter()
logger = get_logger(__name__)

@router.post("/recognize")
async def recognize_handwriting(file: UploadFile = File(...), model_type: str = 'ensemble'):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        recognized_text, confidence = recognize_text(contents, model_type)
        logger.info(f"Successfully recognized text from uploaded image using {model_type} model")
        return {"text": recognized_text, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during text recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_base64")
async def recognize_handwriting_base64(image_data: str = Form(...), model_type: str = 'ensemble'):
    try:
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image data")
        
        recognized_text, confidence = recognize_text(image_data, model_type)
        logger.info(f"Successfully recognized text from base64 image using {model_type} model")
        return {"text": recognized_text, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during text recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_multiline")
async def recognize_multiline_handwriting(file: UploadFile = File(...), model_type: str = 'ensemble'):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        recognized_text, confidence = recognize_text(contents, model_type)
        lines = recognized_text.split('\n')
        logger.info(f"Successfully recognized multi-line text from uploaded image using {model_type} model")
        return {"lines": lines, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during multi-line text recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_realtime")
async def recognize_realtime(file: UploadFile = File(...), model_type: str = 'ensemble'):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        recognized_text, confidence = recognize_text(contents, model_type)
        logger.info(f"Successfully recognized text in real-time using {model_type} model")
        return {"text": recognized_text, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during real-time text recognition: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))