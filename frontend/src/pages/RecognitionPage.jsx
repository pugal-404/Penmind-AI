import React, { useState, useRef } from 'react';
import axios from 'axios';
import ImageUpload from '../components/ImageUpload';
import RecognizedText from '../components/RecognisedText';
import ExportOptions from '../components/ExportOptions';

const RecognitionPage = () => {
  const [recognizedText, setRecognizedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleRecognition = async (file) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setRecognizedText(response.data.text);
    } catch (error) {
      console.error('Error:', error);
      setError('An error occurred during recognition. Please try again.');
      setRecognizedText('');
    } finally {
      setIsLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Unable to access camera. Please check your permissions.');
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      canvasRef.current.toBlob((blob) => {
        handleRecognition(blob);
      }, 'image/jpeg');
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Handwriting Recognition</h1>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Upload Image</h2>
        <ImageUpload onUpload={handleRecognition} />
      </div>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Real-time Recognition</h2>
        <button
          onClick={startCamera}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition duration-300 mr-4"
        >
          Start Camera
        </button>
        <button
          onClick={captureImage}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition duration-300"
        >
          Capture and Recognize
        </button>
        <div className="mt-4">
          <video ref={videoRef} autoPlay playsInline muted className="w-full max-w-md" />
          <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
        </div>
      </div>
      {isLoading && <p className="text-center mt-4">Processing image...</p>}
      {error && <p className="text-center mt-4 text-red-500">{error}</p>}
      {recognizedText && (
        <>
          <RecognizedText text={recognizedText} />
          <ExportOptions text={recognizedText} />
        </>
      )}
    </div>
  );
};

export default RecognitionPage;