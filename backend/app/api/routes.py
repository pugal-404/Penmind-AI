from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.ocr_service import recognize_text
from app.utils.logger import logger

router = APIRouter()

@router.post("/recognize")
async def recognize_handwriting(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        recognized_text = recognize_text(contents)
        logger.info(f"Successfully recognized text from uploaded image")
        return {"text": recognized_text}
    except Exception as e:
        logger.error(f"Error during text recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during recognition")