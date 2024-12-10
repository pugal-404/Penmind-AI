import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { I18nextProvider } from 'react-i18next';
import i18n from './i18n';
import { AnimatePresence } from 'framer-motion';
import HomePage from './pages/HomePage';
import RecognitionPage from './pages/RecognitionPage';
import SettingsPage from './pages/SettingsPage';
import Header from './components/Header';
import Footer from './components/Footer';
import AccessibilityProvider from './contexts/AccessibilityContext';
import './styles/globals.css';
function App() {
  return (
    <ThemeProvider>
      <I18nextProvider i18n={i18n}>
        <AccessibilityProvider>
          <Router>
            <div className="flex flex-col min-h-screen bg-background text-foreground text-adjustable">
              <a href="#main-content" className="skip-link">
                Skip to main content
              </a>
              <Header />
              <AnimatePresence mode="wait">
                <main id="main-content" className="flex-grow z-10">
                  <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/recognition" element={<RecognitionPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                  </Routes>
                </main>
              </AnimatePresence>
              <Footer />
            </div>
          </Router>
        </AccessibilityProvider>
      </I18nextProvider>
    </ThemeProvider>
  );
}
export default App;