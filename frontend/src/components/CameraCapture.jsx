import React, { useRef, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from "./ui/button"
import { Card, CardContent } from "./ui/card"
import { Camera, StopCircle, RefreshCw, CheckCircle } from 'lucide-react'

const CameraCapture = ({ onCapture }) => {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [stream, setStream] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);

  const startCapture = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      setIsCapturing(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  }, []);

  const stopCapture = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    setIsCapturing(false);
    setStream(null);
  }, [stream]);

  const captureImage = useCallback(() => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
      canvas.toBlob(blob => {
        setCapturedImage(URL.createObjectURL(blob));
        onCapture(blob);
        stopCapture();
      }, 'image/jpeg');
    }
  }, [onCapture, stopCapture]);

  const retakePhoto = useCallback(() => {
    setCapturedImage(null);
    startCapture();
  }, [startCapture]);

  return (
    <Card>
      <CardContent className="p-6">
        <motion.div 
          className="aspect-video bg-gray-200 mb-4 rounded-lg overflow-hidden"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <AnimatePresence mode="wait">
            {capturedImage ? (
              <motion.img
                key="captured"
                src={capturedImage}
                alt="Captured"
                className="w-full h-full object-cover"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              />
            ) : (
              <motion.video
                key="video"
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              />
            )}
          </AnimatePresence>
        </motion.div>
        <div className="flex justify-center space-x-4">
          <AnimatePresence mode="wait">
            {!isCapturing && !capturedImage && (
              <motion.div
                key="start"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Button onClick={startCapture}>
                  <Camera className="mr-2 h-4 w-4" /> {t('startCamera')}
                </Button>
              </motion.div>
            )}
            {isCapturing && (
              <motion.div
                key="capturing"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="space-x-4"
              >
                <Button onClick={captureImage} variant="outline">
                  <CheckCircle className="mr-2 h-4 w-4" /> {t('captureImage')}
                </Button>
                <Button onClick={stopCapture} variant="destructive">
                  <StopCircle className="mr-2 h-4 w-4" /> {t('stopCamera')}
                </Button>
              </motion.div>
            )}
            {capturedImage && (
              <motion.div
                key="retake"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Button onClick={retakePhoto}>
                  <RefreshCw className="mr-2 h-4 w-4" /> {t('retakePhoto')}
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </CardContent>
    </Card>
  );
};

export default CameraCapture;

