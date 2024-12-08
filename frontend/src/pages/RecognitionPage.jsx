import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert"
import { Loader2 } from 'lucide-react'
import ImageUpload from '../components/ImageUpload';
import RecognizedText from '../components/RecognizedText';
import ExportOptions from '../components/ExportOptions';
import InteractiveCorrection from '../components/InteractiveCorrection';
import CameraCapture from '../components/CameraCapture';

const RecognitionPage = () => {
  const [recognizedText, setRecognizedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelType, setModelType] = useState('ensemble');
  const [confidence, setConfidence] = useState(null);

  const handleRecognition = useCallback(async (file) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_type', modelType);

      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds timeout
      });

      setRecognizedText(response.data.text);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error('Error:', error);
      if (error.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again.');
      } else if (error.response) {
        setError(`Server error: ${error.response.data.detail || 'Unknown error'}`);
      } else if (error.request) {
        setError('Unable to connect to the server. Please check if the backend is running and accessible.');
      } else {
        setError('An error occurred during recognition. Please try again.');
      }
      setRecognizedText('');
      setConfidence(null);
    } finally {
      setIsLoading(false);
    }
  }, [modelType]);

  const handleRealTimeCapture = useCallback(async (imageBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', imageBlob);
      formData.append('model_type', modelType);

      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 10000, // 10 seconds timeout for real-time recognition
      });

      setRecognizedText(response.data.text);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error('Real-time recognition error:', error);
      // Don't set error state for real-time recognition to avoid disrupting the UI
    }
  }, [modelType]);

  const handleTextCorrection = useCallback((correctedText) => {
    setRecognizedText(correctedText);
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Advanced Handwriting Recognition</h1>
      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload">Image Upload</TabsTrigger>
          <TabsTrigger value="camera">Camera Capture</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
            </CardHeader>
            <CardContent>
              <ImageUpload onUpload={handleRecognition} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="camera">
          <Card>
            <CardHeader>
              <CardTitle>Camera Capture</CardTitle>
            </CardHeader>
            <CardContent>
              <CameraCapture onCapture={handleRecognition} onRealTimeCapture={handleRealTimeCapture} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Model Selection</CardTitle>
        </CardHeader>
        <CardContent>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="ensemble">Ensemble (Default)</option>
            <option value="base">TrOCR Base</option>
            <option value="small">TrOCR Small</option>
            <option value="math">TrOCR Math</option>
          </select>
        </CardContent>
      </Card>
      {isLoading && (
        <div className="flex justify-center items-center mt-8">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          <span>Processing image...</span>
        </div>
      )}
      {error && (
        <Alert variant="destructive" className="mt-8">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {recognizedText && (
        <>
          <RecognizedText text={recognizedText} confidence={confidence} />
          <InteractiveCorrection text={recognizedText} onCorrection={handleTextCorrection} />
          <ExportOptions text={recognizedText} />
        </>
      )}
    </div>
  );
};

export default RecognitionPage;

