import React from 'react';
import { useTranslation } from 'react-i18next';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Volume2 } from 'lucide-react';

const InstructionCard = () => {
  const { t } = useTranslation();
  const { speakText } = useAccessibility();

  const instructions = [
    "Click 'Image Upload' to select an existing image.",
    "Click 'Camera Capture' to take a picture using your device.",
    "Wait for the text to be recognized and displayed below.",
    "Note : This model is evolving and may err. Use Interactive Correction if needed."
  ];

  const handleSpeakInstructions = () => {
    speakText(instructions.join('. '));
  };

  return (
    <Card className="mb-8">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>{t('Instructions')}</CardTitle>
        <Button variant="ghost" size="sm" onClick={handleSpeakInstructions}>
          <Volume2 className="h-4 w-4" />
          <span className="sr-only">{t('speakInstructions')}</span>
        </Button>
      </CardHeader>
      <CardContent>
        <ol className="list-decimal list-inside space-y-2">
          {instructions.map((instruction, index) => (
            <li key={index}>{instruction}</li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
};

export default InstructionCard;
