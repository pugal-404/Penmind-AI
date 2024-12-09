import React from 'react';
import { useTranslation } from 'react-i18next';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Slider } from "./ui/slider"
import { Switch } from "./ui/switch"
import { Label } from "./ui/label"
import { Button } from "./ui/button"
import { Volume2, Type, Contrast, Keyboard, Eye } from 'lucide-react';

const AccessibilityOptions = () => {
  const { t } = useTranslation();
  const { 
    fontSize, 
    setFontSize, 
    highContrast, 
    setHighContrast, 
    textToSpeech, 
    setTextToSpeech,
    keyboardNavigation,
    setKeyboardNavigation,
    screenReaderMode,
    setScreenReaderMode,
    speakText
  } = useAccessibility();

  const handleFontSizeChange = (value) => {
    setFontSize(value[0]);
  };

  const handleHighContrastChange = (checked) => {
    setHighContrast(checked);
  };

  const handleTextToSpeechChange = (checked) => {
    setTextToSpeech(checked);
  };

  const handleKeyboardNavigationChange = (checked) => {
    setKeyboardNavigation(checked);
  };

  const handleScreenReaderModeChange = (checked) => {
    setScreenReaderMode(checked);
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>{t('accessibilityOptions')}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="font-size" className="flex items-center">
              <Type className="mr-2" aria-hidden="true" />
              {t('fontSize')}: {fontSize}px
            </Label>
            <Slider
              id="font-size"
              min={12}
              max={24}
              step={1}
              value={[fontSize]}
              onValueChange={handleFontSizeChange}
              aria-label={t('adjustFontSize')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="high-contrast" className="flex items-center">
              <Contrast className="mr-2" aria-hidden="true" />
              {t('highContrastMode')}
            </Label>
            <Switch
              id="high-contrast"
              checked={highContrast}
              onCheckedChange={handleHighContrastChange}
              aria-label={t('toggleHighContrast')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="text-to-speech" className="flex items-center">
              <Volume2 className="mr-2" aria-hidden="true" />
              {t('Text to Speech')}
            </Label>
            <Switch
              id="text-to-speech"
              checked={textToSpeech}
              onCheckedChange={handleTextToSpeechChange}
              aria-label={t('toggleTextToSpeech')}
            />
          </div>
          {textToSpeech && (
            <Button 
              onClick={() => speakText(t('TextToSpeech'))} 
              className="w-full"
              aria-label={t('testTextToSpeech')}
            >
              <Volume2 className="mr-2 h-4 w-4" aria-hidden="true" />
              {t('TextToSpeech')}
            </Button>
          )}
          <div className="flex items-center justify-between">
            <Label htmlFor="keyboard-navigation" className="flex items-center">
              <Keyboard className="mr-2" aria-hidden="true" />
              {t('Keyboard Navigation')}
            </Label>
            <Switch
              id="keyboard-navigation"
              checked={keyboardNavigation}
              onCheckedChange={handleKeyboardNavigationChange}
              aria-label={t('toggleKeyboardNavigation')}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="screen-reader-mode" className="flex items-center">
              <Eye className="mr-2" aria-hidden="true" />
              {t('Screenreader Mode')}
            </Label>
            <Switch
              id="screen-reader-mode"
              checked={screenReaderMode}
              onCheckedChange={handleScreenReaderModeChange}
              aria-label={t('toggleScreenReaderMode')}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AccessibilityOptions;

