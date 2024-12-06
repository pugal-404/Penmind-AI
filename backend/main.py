from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ml.inference.predict import load_model, predict
from ml.training.train import generate_character_set
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and character set
model = load_model("path/to/your/saved/model.h5")
character_set = generate_character_set()

@app.post("/recognize")
async def recognize_handwriting(file: UploadFile = File(...)):
    contents = await file.read()
    image = io.BytesIO(contents)
    
    recognized_text = predict(model, image, character_set)
    
    return {"text": recognized_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)