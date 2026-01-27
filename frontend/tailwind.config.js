/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        triage: {
          red: "#ef4444",
          orange: "#f97316",
          yellow: "#eab308",
          green: "#22c55e",
          blue: "#3b82f6",
        },
      },
    },
  },
  plugins: [],
};
