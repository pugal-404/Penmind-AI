import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Textarea } from "./ui/textarea"

const InteractiveCorrection = ({ text, onCorrection }) => {
  const [correctedText, setCorrectedText] = useState(text);

  useEffect(() => {
    setCorrectedText(text);
  }, [text]);

  const handleChange = (e) => {
    setCorrectedText(e.target.value);
    onCorrection(e.target.value);
  };

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Interactive Correction</CardTitle>
      </CardHeader>
      <CardContent>
        <Textarea
          value={correctedText}
          onChange={handleChange}
          className="w-full h-40 p-2"
          placeholder="Edit the recognized text here..."
        />
      </CardContent>
    </Card>
  );
};

export default InteractiveCorrection;

