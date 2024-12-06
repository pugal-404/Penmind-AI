import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="text-center">
      <h1 className="text-4xl font-bold mb-6">Welcome to Handwriting Recognition</h1>
      <p className="mb-8">
        Our advanced system helps convert handwritten text into digital format with high accuracy.
      </p>
      <Link
        to="/recognition"
        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition duration-300"
      >
        Start Recognition
      </Link>
    </div>
  );
};

export default HomePage;

