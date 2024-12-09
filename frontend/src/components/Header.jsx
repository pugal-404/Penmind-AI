import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';
import { motion } from 'framer-motion';
import { Sun, Moon, Home, FileText, Settings } from 'lucide-react';
import { Button } from './ui/button';

const Header = () => {
  const { t } = useTranslation();
  const { theme, toggleTheme } = useTheme();

  const iconVariants = {
    hover: { scale: 1.2, rotate: 10, transition: { duration: 0.3 } }
  };

  return (
    <header className="bg-gradient-to-r from-primary to-secondary text-primary-foreground z-10 sticky top-0">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link to="/" className="flex items-center" aria-label={t('home')}>
          <motion.div
            className="text-2xl font-bold tracking-tighter"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
              Penmind AI
          </motion.div>
        </Link>
        <nav>
          <ul className="flex space-x-6">
            <li>
              <Link to="/" aria-label={t('home')}>
                <motion.div variants={iconVariants} whileHover="hover">
                  <Home className="h-6 w-6" />
                </motion.div>
              </Link>
            </li>
            <li>
              <Link to="/recognition" aria-label={t('recognition')}>
                <motion.div variants={iconVariants} whileHover="hover">
                  <FileText className="h-6 w-6" />
                </motion.div>
              </Link>
            </li>
            <li>
              <Link to="/settings" aria-label={t('settings')}>
                <motion.div variants={iconVariants} whileHover="hover">
                  <Settings className="h-6 w-6" />
                </motion.div>
              </Link>
            </li>
          </ul>
        </nav>
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? t('switchToLightMode') : t('switchToDarkMode')}
        >
          <motion.div variants={iconVariants} whileHover="hover">
            {theme === 'dark' ? <Sun className="h-6 w-6" /> : <Moon className="h-6 w-6" />}
          </motion.div>
        </Button>
      </div>
    </header>
  );
};

export default Header;