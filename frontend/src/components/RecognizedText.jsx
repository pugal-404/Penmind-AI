import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"

const RecognizedText = ({ text, confidence }) => {
  const lines = text.split('\n');

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Recognized Text</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="bg-gray-100 p-4 rounded-lg">
          {lines.map((line, index) => (
            <p key={index} className="whitespace-pre-wrap mb-2 text-lg">
              {line}
            </p>
          ))}
        </div>
        {confidence !== null && (
          <p className="mt-2 text-sm text-gray-600">
            Confidence: {(confidence * 100).toFixed(2)}%
          </p>
        )}
      </CardContent>
    </Card>
  );
};

export default RecognizedText;

