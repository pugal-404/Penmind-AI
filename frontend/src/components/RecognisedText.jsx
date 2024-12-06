import React from 'react';

const RecognizedText = ({ text }) => {
  return (
    <div className="mt-8">
      <h2 className="text-2xl font-bold mb-4">Recognized Text</h2>
      <div className="bg-gray-100 p-4 rounded-lg">
        <p className="whitespace-pre-wrap">{text}</p>
      </div>
    </div>
  );
};

export default RecognizedText;