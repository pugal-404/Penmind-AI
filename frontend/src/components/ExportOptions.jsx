import React from 'react';
import { Button } from "./ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Copy, FileText, FileJson, FileIcon as FileWord, FileIcon as FilePdf } from 'lucide-react'

const ExportOptions = ({ text }) => {
  const copyToClipboard = () => {
    navigator.clipboard.writeText(text).then(() => {
      alert('Text copied to clipboard!');
    });
  };

  const exportAsTxt = () => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recognized_text.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsJson = () => {
    const jsonData = JSON.stringify({ recognizedText: text });
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recognized_text.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsDocx = () => {
    // This is a simplified example. In a real-world scenario, you'd use a library like docx.js
    const blob = new Blob([text], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recognized_text.docx';
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsPdf = () => {
    // This is a simplified example. In a real-world scenario, you'd use a library like jsPDF
    const blob = new Blob([text], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recognized_text.pdf';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Export Options</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-4">
        <Button onClick={copyToClipboard}>
          <Copy className="mr-2 h-4 w-4" /> Copy to Clipboard
        </Button>
        <Button onClick={exportAsTxt}>
          <FileText className="mr-2 h-4 w-4" /> Export as TXT
        </Button>
        <Button onClick={exportAsJson}>
          <FileJson className="mr-2 h-4 w-4" /> Export as JSON
        </Button>
        <Button onClick={exportAsDocx}>
          <FileWord className="mr-2 h-4 w-4" /> Export as DOCX
        </Button>
        <Button onClick={exportAsPdf}>
          <FilePdf className="mr-2 h-4 w-4" /> Export as PDF
        </Button>
      </CardContent>
    </Card>
  );
};

export default ExportOptions;

