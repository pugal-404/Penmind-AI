import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="bg-blue-600 text-white">
      <div className="container mx-auto px-4 py-6">
        <nav className="flex justify-between items-center">
          <Link to="/" className="text-2xl font-bold">
            Handwriting Recognition
          </Link>
          <ul className="flex space-x-4">
            <li>
              <Link to="/" className="hover:underline">
                Home
              </Link>
            </li>
            <li>
              <Link to="/recognition" className="hover:underline">
                Recognition
              </Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;

