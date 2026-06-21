"use client";

import { motion, useReducedMotion } from "framer-motion";

export default function Template({
  children,
}: {
  children: React.ReactNode;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 18 }}
      animate={reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
      transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
      className="flex-1"
    >
      {children}
    </motion.div>
  );
}
