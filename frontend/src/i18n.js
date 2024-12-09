import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        translation: {
          handwritingRecognition: 'Handwriting Recognition',
          imageUpload: 'Image Upload',
          cameraCapture: 'Camera Capture',
          uploadImage: 'Upload Image',
          processingImage: 'Processing image...',
          error: 'Error',
          recognizedText: 'Recognized Text',
          confidence: 'Confidence',
          processingTime: 'Processing Time',
          modelUsed: 'Model Used',
          seconds: 'seconds',
          startCamera: 'Start Camera',
          captureImage: 'Capture Image',
          stopCamera: 'Stop Camera',
          retakePhoto: 'Retake Photo',
          dropImageHere: 'Drop the image here',
          dragDropImage: "Drag 'n' drop an image here, or click to select a file",
          imageUploaded: 'Image uploaded',
          accessibilityOptions: 'Accessibility Options',
          fontSize: 'Font Size',
          highContrastMode: 'High Contrast Mode',
          textToSpeech: 'Text-to-Speech',
          exportOptions: 'Export Options',
          copyToClipboard: 'Copy to Clipboard',
          exportAsTxt: 'Export as TXT',
          exportAsDocx: 'Export as DOCX',
          exportAsPdf: 'Export as PDF',
          exportAsJson: 'Export as JSON',
          copiedToClipboard: 'Copied to clipboard!',
          recognitionError: 'An error occurred during recognition. Please try again.',
        },
      },
      // Add more languages here
    },
    lng: 'en', // Set the default language
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;

