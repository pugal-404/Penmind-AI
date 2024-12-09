import React from 'react';
import { motion } from 'framer-motion';
import { GitlabIcon as GitHub, Twitter, Mail, Info } from 'lucide-react';

const Footer = () => {
  const iconVariants = {
    hover: { scale: 1.2, rotate: 10, transition: { duration: 0.3 } }
  };

  return (
    <footer className="bg-gradient-to-r from-primary to-secondary text-primary-foreground mt-auto">
      <div className="container mx-auto px-4 py-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
        <p>&copy; {new Date().getFullYear()} Penmind AI. All rights reserved.</p>
          <nav className="mt-4 md:mt-0">
            <ul className="flex space-x-6">
              <li>
                <a href="https://github.com/aiwriter" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                  <motion.div variants={iconVariants} whileHover="hover">
                    <GitHub className="h-6 w-6" />
                  </motion.div>
                </a>
              </li>
              <li>
                <a href="https://twitter.com/aiwriter" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
                  <motion.div variants={iconVariants} whileHover="hover">
                    <Twitter className="h-6 w-6" />
                  </motion.div>
                </a>
              </li>
              <li>
                <a href="mailto:contact@aiwriter.com" aria-label="Email">
                  <motion.div variants={iconVariants} whileHover="hover">
                    <Mail className="h-6 w-6" />
                  </motion.div>
                </a>
              </li>
              <li>
                <a href="/about" aria-label="About">
                  <motion.div variants={iconVariants} whileHover="hover">
                    <Info className="h-6 w-6" />
                  </motion.div>
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </footer>
  );
};

export default Footer;