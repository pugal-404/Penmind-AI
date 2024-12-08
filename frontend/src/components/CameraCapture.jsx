import React, { useRef, useState, useCallback, useEffect } from "react";
import axios from "axios";
import { Button } from "./ui/button";
import { Camera, RefreshCw, Play, Pause } from 'lucide-react';
import ExportOptions from './ExportOptions'; // Assuming ExportOptions is imported

const CameraCapture = ({ onCapture, onRealTimeCapture }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [recognizedText, setRecognizedText] = useState(null);
  const [isRealTimeMode, setIsRealTimeMode] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
      setIsCameraOn(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
      alert("Unable to access the camera. Please check permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    const stream = videoRef.current.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    videoRef.current.srcObject = null;
    setIsCameraOn(false);
    setIsRealTimeMode(false);
  }, []);

  const captureImage = async () => {
    setIsLoading(true);
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (canvas && video) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (blob) {
          // Send captured image to backend for recognition
          try {
            const formData = new FormData();
            formData.append("file", blob);
            const response = await axios.post("http://localhost:8000/recognize_realtime", formData, {
              timeout: 10000 // 10 seconds timeout
            });
            setCapturedImage(URL.createObjectURL(blob));
            setRecognizedText(response.data.recognized_text); // Update recognized text state
            onCapture(response.data.recognized_text); // Pass recognized text to parent
          } catch (error) {
            console.error("Error during recognition:", error);
            alert("Failed to process the image. Please try again.");
          } finally {
            setIsLoading(false);
          }
        }
      }, "image/jpeg");
    }
    stopCamera();
  };

  const toggleRealTimeMode = useCallback(() => {
    setIsRealTimeMode((prev) => !prev);
  }, []);

  useEffect(() => {
    let interval;
    if (isRealTimeMode && isCameraOn) {
      interval = setInterval(() => {
        if (videoRef.current && canvasRef.current) {
          const context = canvasRef.current.getContext('2d');
          context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
          canvasRef.current.toBlob((blob) => {
            onRealTimeCapture(blob); // Send to parent component
            recognizeText(blob); // Real-time recognition
          }, 'image/jpeg');
        }
      }, 1000); // Capture every second
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRealTimeMode, isCameraOn, onRealTimeCapture]);

  const recognizeText = async (imageBlob) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', imageBlob);

    try {
      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setRecognizedText(response.data.recognized_text); // Save recognized text
    } catch (err) {
      console.error('Error during text recognition:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const retakePhoto = useCallback(() => {
    setCapturedImage(null);
    setRecognizedText(null);
    startCamera();
  }, [startCamera]);

  return (
    <div className="camera-capture">
      <div className="status-badge">
        {isCameraOn ? (
          <span className="badge badge-green">Camera On</span>
        ) : (
          <span className="badge badge-red">Camera Off</span>
        )}
      </div>
      <div className="video-container">
        <video ref={videoRef} className="video-feed"></video>
        <canvas ref={canvasRef} className="hidden"></canvas>
      </div>
      <div className="controls">
        {!isCameraOn ? (
          <Button onClick={startCamera} className="w-full">
            <Camera className="mr-2 h-4 w-4" /> Start Camera
          </Button>
        ) : (
          <Button onClick={stopCamera} className="btn btn-danger">
            Stop Camera
          </Button>
        )}
        <Button
          onClick={captureImage}
          className="btn btn-success"
          disabled={!isCameraOn || isLoading}
        >
          {isLoading ? "Processing..." : "Capture Image"}
        </Button>
        <Button onClick={toggleRealTimeMode} className="btn btn-secondary">
          {isRealTimeMode ? (
            <Pause className="mr-2 h-4 w-4" />
          ) : (
            <Play className="mr-2 h-4 w-4" />
          )}
          {isRealTimeMode ? "Stop Real-time" : "Start Real-time"}
        </Button>
      </div>
      {capturedImage && (
        <div className="captured-image-container">
          <img src={capturedImage} alt="Captured" className="w-full h-auto" />
          <Button
            onClick={retakePhoto}
            className="absolute bottom-4 left-1/2 transform -translate-x-1/2"
          >
            <RefreshCw className="mr-2 h-4 w-4" /> Retake Photo
          </Button>
        </div>
      )}
      {recognizedText && !isLoading && (
        <div className="mt-4">
          <h3>Recognized Text:</h3>
          <p>{recognizedText}</p>
          <ExportOptions recognizedText={recognizedText} /> {/* Export options for DOCX, PDF */}
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
