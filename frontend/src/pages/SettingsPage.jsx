import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { useTheme } from '../contexts/ThemeContext';
import AccessibilityOptions from '../components/AccessibilityOptions';
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Sun, Moon, Zap } from 'lucide-react';

const SettingsPage = () => {
  const { t } = useTranslation();
  const { theme, setTheme } = useTheme();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 settings-page">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.h1 className="text-3xl font-bold mb-6" variants={itemVariants}>{t('Settings')}</motion.h1>
        <motion.div variants={itemVariants}>
          <AccessibilityOptions />
        </motion.div>
        <motion.div variants={itemVariants}>
          <Card>
            <CardHeader>
              <CardTitle>{t('Custom Themes')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-4">
                <Button 
                  onClick={() => setTheme('light')} 
                  className={`${theme === 'light' ? 'bg-primary' : 'bg-secondary'}`}
                  aria-label={t('lightTheme')}
                >
                  <Sun className="mr-2 h-4 w-4" aria-hidden="true" />
                  {t('Light')}
                </Button>
                <Button 
                  onClick={() => setTheme('dark')} 
                  className={`${theme === 'dark' ? 'bg-primary' : 'bg-secondary'}`}
                  aria-label={t('darkTheme')}
                >
                  <Moon className="mr-2 h-4 w-4" aria-hidden="true" />
                  {t('Dark')}
                </Button>
                <Button 
                  onClick={() => setTheme('futuristic')} 
                  className={`${theme === 'futuristic' ? 'bg-primary' : 'bg-secondary'}`}
                  aria-label={t('futuristicTheme')}
                >
                  <Zap className="mr-2 h-4 w-4" aria-hidden="true" />
                  {t('Standard')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default SettingsPage;

