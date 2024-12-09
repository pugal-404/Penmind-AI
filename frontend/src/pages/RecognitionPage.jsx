import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import axios from 'axios';
import { Button } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert"
import { Loader2, Upload, Camera, Volume2 } from 'lucide-react'
import ImageUpload from '../components/ImageUpload';
import CameraCapture from '../components/CameraCapture';
import RecognizedText from '../components/RecognizedText';
import ExportOptions from '../components/ExportOptions';
import InstructionCard from '../components/InstructionCard';
import { useAccessibility } from '../contexts/AccessibilityContext';
const RecognitionPage = () => {
  const { t } = useTranslation();
  const [recognizedText, setRecognizedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);
  const [modelUsed, setModelUsed] = useState('');
  const { speakText } = useAccessibility();
  const handleRecognition = useCallback(async (file, isRealTime = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const endpoint = isRealTime ? '/recognize_realtime' : '/recognize';
      const response = await axios.post(`http://localhost:8000${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: isRealTime ? 10000 : 60000,
      });
      setRecognizedText(response.data.text);
      setConfidence(response.data.confidence);
      setProcessingTime(response.data.processing_time);
      setModelUsed(response.data.model_used);
    } catch (error) {
      console.error('Error:', error);
      setError(t('recognitionError'));
    } finally {
      setIsLoading(false);
    }
  }, [t]);
  const handleSpeakRecognizedText = () => {
    speakText(recognizedText);
  };
  return (
    <div className="container mx-auto px-4 py-8">
      <InstructionCard />
      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload">
            <Upload className="mr-2 h-4 w-4" />
            {t('imageUpload')}
          </TabsTrigger>
          <TabsTrigger value="camera">
            <Camera className="mr-2 h-4 w-4" />
            {t('cameraCapture')}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>{t('uploadImage')}</CardTitle>
            </CardHeader>
            <CardContent>
              <ImageUpload onUpload={(file) => handleRecognition(file, false)} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="camera">
          <Card>
            <CardHeader>
              <CardTitle>{t('cameraCapture')}</CardTitle>
            </CardHeader>
            <CardContent>
              <CameraCapture onCapture={(file) => handleRecognition(file, true)} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="flex justify-center items-center mt-8"
        >
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          <span>{t('processingImage')}</span>
        </motion.div>
      )}
      {error && (
        <Alert variant="destructive" className="mt-8">
          <AlertTitle>{t('error')}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {recognizedText && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <RecognizedText 
            text={recognizedText} 
            confidence={confidence}
            processingTime={processingTime}
            modelUsed={modelUsed}
          />
          <Button onClick={handleSpeakRecognizedText} className="mt-4">
            <Volume2 className="mr-2 h-4 w-4" />
            {t('speakRecognizedText')}
          </Button>
          <ExportOptions text={recognizedText} />
        </motion.div>
      )}
    </div>
  );
};
export default RecognitionPage;