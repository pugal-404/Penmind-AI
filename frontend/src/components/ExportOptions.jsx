import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { Button } from "./ui/button"
import { Card, CardContent, CardHeader, CardTitle } from"./ui/card"
import { Copy, FileText, FileJson, FileIcon as FileWord, FileIcon as FilePdf } from 'lucide-react'
import { saveAs } from 'file-saver';
import { Document, Packer, Paragraph } from 'docx';
import { jsPDF } from 'jspdf';

const ExportOptions = ({ text }) => {
  const { t } = useTranslation();

  const copyToClipboard = () => {
    navigator.clipboard.writeText(text).then(() => {
      alert(t('copiedToClipboard'));
    });
  };

  const exportAsTxt = () => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    saveAs(blob, 'recognized_text.txt');
  };

  const exportAsDocx = () => {
    const doc = new Document({
      sections: [{
        properties: {},
        children: [
          new Paragraph(text),
        ],
      }],
    });

    Packer.toBlob(doc).then(blob => {
      saveAs(blob, 'recognized_text.docx');
    });
  };

  const exportAsPdf = () => {
    const pdf = new jsPDF();
    pdf.text(text, 10, 10);
    pdf.save('recognized_text.pdf');
  };

  const exportAsJson = () => {
    const jsonData = JSON.stringify({ recognizedText: text });
    const blob = new Blob([jsonData], { type: 'application/json' });
    saveAs(blob, 'recognized_text.json');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>{t('exportOptions')}</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Button onClick={copyToClipboard} className="w-full">
            <Copy className="mr-2 h-4 w-4" /> {t('copyToClipboard')}
          </Button>
          <Button onClick={exportAsTxt} className="w-full">
            <FileText className="mr-2 h-4 w-4" /> {t('exportAsTxt')}
          </Button>
          <Button onClick={exportAsDocx} className="w-full">
            <FileWord className="mr-2 h-4 w-4" /> {t('exportAsDocx')}
          </Button>
          <Button onClick={exportAsPdf} className="w-full">
            <FilePdf className="mr-2 h-4 w-4" /> {t('exportAsPdf')}
          </Button>
          <Button onClick={exportAsJson} className="w-full">
            <FileJson className="mr-2 h-4 w-4" /> {t('exportAsJson')}
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default ExportOptions;

