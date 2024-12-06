import React from 'react';
import { saveAs } from 'file-saver';
import { Document, Packer, Paragraph } from 'docx';
import { jsPDF } from 'jspdf';

const ExportOptions = ({ text }) => {
  const exportAsTxt = () => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    saveAs(blob, 'recognized_text.txt');
  };

  const exportAsDocx = () => {
    const doc = new Document({
      sections: [{
        properties: {},
        children: [
          new Paragraph({
            children: [{ text: text }],
          }),
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
    const blob = new Blob([jsonData], { type: 'application/json;charset=utf-8' });
    saveAs(blob, 'recognized_text.json');
  };

  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-4">Export Options</h2>
      <div className="space-x-4">
        <button
          onClick={exportAsTxt}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition duration-300"
        >
          Export as TXT
        </button>
        <button
          onClick={exportAsDocx}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition duration-300"
        >
          Export as DOCX
        </button>
        <button
          onClick={exportAsPdf}
          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition duration-300"
        >
          Export as PDF
        </button>
        <button
          onClick={exportAsJson}
          className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 transition duration-300"
        >
          Export as JSON
        </button>
      </div>
    </div>
  );
};

export default ExportOptions;