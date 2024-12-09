import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { Upload, Brain, Download } from 'lucide-react';

const HomePage = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  return (
    <div className="container mx-auto px-4 py-16 select-none">
      <motion.div
        className="text-center"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.h1 
      className="text-2xl md:text-3xl lg:text-4xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary"
      variants={itemVariants}
        >
       AI-Powered Handwriting Recognition
      </motion.h1>

        <motion.p 
          className="text-lg md:text-xl mb-12 max-w-2xl mx-auto"
          variants={itemVariants}
        >
          Experience the future of handwriting digitization with our advanced machine learning technology.
        </motion.p>
        <motion.div variants={itemVariants}>
          <Link to="/recognition">
            <Button size="lg" className="text-lg px-8 py-6 bg-gradient-to-r from-primary to-secondary hover:from-primary/80 hover:to-secondary/80 transition-all duration-300 transform hover:scale-105">
              Start Recognition
            </Button>
          </Link>
        </motion.div>
      </motion.div>

      <motion.div 
        className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {[
          { title: "Upload", description: "Easily upload handwritten documents", icon: Upload },
          { title: "Recognize", description: "Advanced AI processes your handwriting", icon: Brain },
          { title: "Export", description: "Download your digitized text instantly", icon: Download }
        ].map((feature, index) => (
          <motion.div key={index} variants={itemVariants}>
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 text-center">
                <feature.icon className="h-12 w-12 mx-auto mb-4 text-primary" />
                <h2 className="text-2xl font-semibold mb-2">{feature.title}</h2>
                <p>{feature.description}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
};

export default HomePage;