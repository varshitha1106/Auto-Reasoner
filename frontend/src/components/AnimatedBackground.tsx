import React, { useEffect, useState } from 'react';
import './AnimatedBackground.css';

interface SymbolData {
  id: number;
  char: string;
  left: string;
  duration: string;
  delay: string;
  size: string;
  drift: string;
  rotation: string;
  color: string;
}

const SYMBOLS = ['{', '}', '(', ')', ';', '</>', '#', '=>', '[]', '*'];
const COLORS = [
  '#00ffff', // Cyan
  '#ff00ff', // Magenta
  '#00ff00', // Lime
  '#ffff00', // Yellow
  '#ff4500', // Orange Red
  '#1e90ff', // Dodger Blue
  '#ff1493', // Deep Pink
  '#7fff00', // Chartreuse
  '#00bfff', // Deep Sky Blue
  '#9400d3', // Dark Violet
];

const AnimatedBackground: React.FC = () => {
  const [symbols, setSymbols] = useState<SymbolData[]>([]);

  useEffect(() => {
    const symbolCount = 25; // Number of floating symbols
    const newSymbols: SymbolData[] = [];

    for (let i = 0; i < symbolCount; i++) {
      newSymbols.push({
        id: i,
        char: SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)],
        left: `${Math.random() * 100}%`,
        duration: `${Math.random() * 15 + 15}s`, // Between 15s and 30s
        delay: `${Math.random() * 20 * -1}s`, // Negative delay to start mid-animation
        size: `${Math.random() * 3 + 3}rem`, // 3rem to 6rem
        drift: `${Math.random() * 10 - 5}vw`, // -5vw to 5vw sideways
        rotation: `${Math.random() * 360}deg`,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
      });
    }
    setSymbols(newSymbols);
  }, []);

  return (
    <div className="animated-background-container" aria-hidden="true">
      {symbols.map((sym) => (
        <span
          key={sym.id}
          className="floating-symbol"
          style={{
            left: sym.left,
            animationDuration: sym.duration,
            animationDelay: sym.delay,
            fontSize: sym.size,
            color: sym.color,
            '--drift': sym.drift,
            '--rotation': sym.rotation,
          } as React.CSSProperties}
        >
          {sym.char}
        </span>
      ))}
    </div>
  );
};

export default AnimatedBackground;
