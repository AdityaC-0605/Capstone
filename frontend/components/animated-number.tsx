"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

interface AnimatedNumberProps {
  value: number;
  duration?: number;
  formatter?: (value: number) => string;
  className?: string;
  startOnView?: boolean;
}

export function AnimatedNumber({
  value,
  duration = 1200,
  formatter = (input) => input.toFixed(2),
  className,
  startOnView = false,
}: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const [canAnimate, setCanAnimate] = useState(!startOnView);
  const reduceMotion = useReducedMotion();
  const spanRef = useRef<HTMLSpanElement | null>(null);
  const valueRef = useRef(0);

  useEffect(() => {
    valueRef.current = displayValue;
  }, [displayValue]);

  useEffect(() => {
    if (!startOnView || reduceMotion) {
      setCanAnimate(true);
      return;
    }

    const node = spanRef.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setCanAnimate(true);
          observer.disconnect();
        }
      },
      { threshold: 0.3 },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [reduceMotion, startOnView]);

  useEffect(() => {
    if (!canAnimate) return;
    if (reduceMotion) {
      setDisplayValue(value);
      return;
    }

    let frame = 0;
    let startTime: number | null = null;
    const startValue = valueRef.current;

    const animate = (timestamp: number) => {
      if (startTime === null) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const eased = 1 - (1 - progress) * (1 - progress);
      setDisplayValue(startValue + (value - startValue) * eased);
      if (progress < 1) {
        frame = window.requestAnimationFrame(animate);
      }
    };

    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, [canAnimate, duration, reduceMotion, value]);

  return (
    <span ref={spanRef} className={className}>
      {formatter(displayValue)}
    </span>
  );
}
