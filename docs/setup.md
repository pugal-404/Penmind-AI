# Setup Guide for Handwriting Recognition System

## Prerequisites

- Python 3.9 or later
- Node.js 14 or later
- Docker and Docker Compose (for containerized deployment)

## Local Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/handwriting-recognition-system.git
   cd handwriting-recognition-system
   ```

2. Set up the backend:
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```
   cd ../frontend
   npm install
   ```

4. Start the backend server:
   ```
   cd ../backend
   uvicorn main:app --reload
   ```

5. Start the frontend development server:
   ```
   cd ../frontend
   npm start
   ```

6. Access the application at `http://localhost:3000`

## Training the Model

1. Prepare your dataset in the `ml/data` directory.
2. Run the training script:
   ```
   cd ml
   python training/train.py
   ```

3. The trained model will be saved as `handwriting_recognition_model.h5`.

## Containerized Deployment

1. Build and run the containers:
   ```
   docker-compose up --build
   ```

2. Access the application at `http://localhost:3000`

For more detailed information on API usage and system architecture, please refer to the `api_docs.md` and `user_guide.md` files in the `docs` directory.