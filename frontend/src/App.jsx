import React from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import RecognitionPage from './pages/RecognitionPage';

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <>
        <Header />
        <HomePage />
        <Footer />
      </>
    ),
  },
  {
    path: "/recognition",
    element: (
      <>
        <Header />
        <RecognitionPage />
        <Footer />
      </>
    ),
  },
], {
  future: {
    v7_startTransition: true,
    v7_relativeSplatPath: true,
    v7_fetcherPersist: true,
    v7_normalizeFormMethod: true,
    v7_partialHydration: true,
    v7_skipActionErrorRevalidation: true,
  },
});

const App = () => {
  return (
    <RouterProvider router={router} />
  );
};

export default App;

