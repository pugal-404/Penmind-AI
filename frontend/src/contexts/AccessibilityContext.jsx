import React, { createContext, useContext, useState, useEffect } from 'react';

const AccessibilityContext = createContext();

export const useAccessibility = () => useContext(AccessibilityContext);

const AccessibilityProvider = ({ children }) => {
  const [fontSize, setFontSize] = useState(() => {
    const saved = localStorage.getItem('fontSize');
    return saved ? parseInt(saved, 10) : 16;
  });
  const [highContrast, setHighContrast] = useState(() => {
    const saved = localStorage.getItem('highContrast');
    return saved === 'true';
  });
  const [textToSpeech, setTextToSpeech] = useState(() => {
    const saved = localStorage.getItem('textToSpeech');
    return saved === 'true';
  });
  const [keyboardNavigation, setKeyboardNavigation] = useState(() => {
    const saved = localStorage.getItem('keyboardNavigation');
    return saved === 'true';
  });
  const [screenReaderMode, setScreenReaderMode] = useState(() => {
    const saved = localStorage.getItem('screenReaderMode');
    return saved === 'true';
  });

  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}px`;
    localStorage.setItem('fontSize', fontSize);
  }, [fontSize]);

  useEffect(() => {
    if (highContrast) {
      document.body.classList.add('high-contrast');
    } else {
      document.body.classList.remove('high-contrast');
    }
    localStorage.setItem('highContrast', highContrast);
  }, [highContrast]);

  useEffect(() => {
    localStorage.setItem('textToSpeech', textToSpeech);
  }, [textToSpeech]);

  useEffect(() => {
    if (keyboardNavigation) {
      document.body.classList.add('keyboard-navigation');
    } else {
      document.body.classList.remove('keyboard-navigation');
    }
    localStorage.setItem('keyboardNavigation', keyboardNavigation);
  }, [keyboardNavigation]);

  useEffect(() => {
    if (screenReaderMode) {
      document.body.setAttribute('aria-live', 'polite');
    } else {
      document.body.removeAttribute('aria-live');
    }
    localStorage.setItem('screenReaderMode', screenReaderMode);
  }, [screenReaderMode]);

  const speakText = (text) => {
    if (textToSpeech && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <AccessibilityContext.Provider
      value={{
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
        speakText,
      }}
    >
      {children}
    </AccessibilityContext.Provider>
  );
};

export default AccessibilityProvider;

