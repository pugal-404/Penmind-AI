import React, { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Volume2, VolumeX } from 'lucide-react';

const InstructionCard = () => {
  const { t } = useTranslation();
  const { textToSpeech } = useAccessibility();
  const [isSpeaking, setIsSpeaking] = useState(false);
  const utteranceRef = useRef(null);

  const instructions = [
    t("Click 'Image Upload' to select an existing image.",),
    t("Click 'Camera Capture' to take a picture using your device.",),
    t("Wait for the text to be recognized and displayed below."),
    t("Note: This model is evolving and may err. Use Interactive Correction if needed.",),
  ];

  const handleSpeakInstructions = () => {
    if (!textToSpeech) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    setIsSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(instructions.join('. '));
    utteranceRef.current = utterance;

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = () => {
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  React.useEffect(() => {
    return () => {
      if (utteranceRef.current) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  return (
    <Card className="mb-8">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>{t('Instructions')}</CardTitle>
        {textToSpeech && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSpeakInstructions}
            aria-label={isSpeaking ? t('stopSpeaking') : t('speakInstructions')}
          >
            {isSpeaking ? (
              <VolumeX className="h-4 w-4" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </Button>
        )}
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

