import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Progress } from "./ui/progress"

const RecognizedText = ({ text, confidence, processingTime, modelUsed }) => {
  const { t } = useTranslation();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>{t('recognizedText')}</CardTitle>
        </CardHeader>
        <CardContent>
          <motion.div 
            className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg mb-4 max-h-60 overflow-y-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <p className="whitespace-pre-wrap">{text}</p>
          </motion.div>
          <div className="space-y-4">
            <div>
              <span className="font-semibold">{t('confidence')}:</span>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <Progress value={confidence * 100} className="mt-2" />
              </motion.div>
              <span className="text-sm text-gray-600">{(confidence * 100).toFixed(2)}%</span>
            </div>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <span className="font-semibold">{t('processingTime')}:</span> {processingTime.toFixed(2)} {t('seconds')}
            </motion.p>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <span className="font-semibold">{t('modelUsed')}:</span> {modelUsed}
            </motion.p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default RecognizedText;

