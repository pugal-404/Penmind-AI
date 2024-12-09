import React, { useEffect, useRef } from 'react';

const BackgroundAnimation = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const examples = [
      { handwritten: "E = mc²", typed: "E = mc²" },
      { handwritten: "AI is the future", typed: "AI is the future" },
      { handwritten: "The quick brown fox", typed: "The quick brown fox" }
    ];
    let currentExample = 0;
    let phase = 'writing';
    let progress = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const example = examples[currentExample];
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      ctx.font = '30px "Comic Sans MS", cursive';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      if (phase === 'writing') {
        const text = example.handwritten.slice(0, Math.floor(progress * example.handwritten.length));
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(text, centerX, centerY);
      } else if (phase === 'scanning') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(example.handwritten, centerX, centerY);
        
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop(0, 'rgba(0, 255, 0, 0)');
        gradient.addColorStop(progress, 'rgba(0, 255, 0, 0.5)');
        gradient.addColorStop(progress + 0.1, 'rgba(0, 255, 0, 0.5)');
        gradient.addColorStop(1, 'rgba(0, 255, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.height, canvas.height);
      } else if (phase === 'transforming') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(example.handwritten, centerX, centerY - 20 * (1 - progress));
        
        ctx.font = '24px Arial, sans-serif';
        ctx.fillStyle = `rgba(255, 255, 255, ${progress})`;
        ctx.fillText(example.typed, centerX, centerY + 20 * progress);
        
        ctx.beginPath();
        ctx.moveTo(centerX - 50, centerY);
        ctx.lineTo(centerX + 50, centerY);
        ctx.lineTo(centerX + 40, centerY - 10);
        ctx.moveTo(centerX + 50, centerY);
        ctx.lineTo(centerX + 40, centerY + 10);
        ctx.strokeStyle = `rgba(255, 255, 255, ${progress})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      progress += 0.005;
      if (progress >= 1) {
        progress = 0;
        if (phase === 'writing') {
          phase = 'scanning';
        } else if (phase === 'scanning') {
          phase = 'transforming';
        } else {
          phase = 'writing';
          currentExample = (currentExample + 1) % examples.length;
        }
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full -z-10" />;
};

export default BackgroundAnimation;

