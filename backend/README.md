# Handwriting Recognition Backend

This is the backend for the Advanced Handwriting Recognition System. It provides an API for processing handwritten text images and returning the recognized text.

## Features

- FastAPI-based REST API
- Integration with machine learning model for text recognition
- Image preprocessing

## Getting Started

1. Install dependencies:
pip install -r requirements.txt

2. Start the server:
uvicorn main:app --reload

3. The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /recognize`: Upload an image file to recognize handwritten text

## Testing

Run the tests using:
pytest


## Learn More

To learn FastAPI, check out the [FastAPI documentation](https://fastapi.tiangolo.com/).