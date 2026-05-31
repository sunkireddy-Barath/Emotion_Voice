/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f4ff',
          100: '#dbe4ff',
          200: '#bac8ff',
          300: '#91a7ff',
          400: '#748ffc',
          500: '#5c7cfa',
          600: '#4c6ef5',
          700: '#4263eb',
          800: '#3b5bdb',
          900: '#364fc7',
        },
        emotion: {
          neutral:      '#6b7280',
          happy:        '#f59e0b',
          sad:          '#3b82f6',
          angry:        '#ef4444',
          excited:      '#f97316',
          fear:         '#8b5cf6',
          surprise:     '#ec4899',
          calm:         '#10b981',
          serious:      '#64748b',
          motivational: '#06b6d4',
          questioning:  '#a78bfa',
          storytelling: '#84cc16',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'wave': 'wave 1.5s ease-in-out infinite',
      },
      keyframes: {
        wave: {
          '0%, 100%': { transform: 'scaleY(1)' },
          '50%':      { transform: 'scaleY(2)' },
        },
      },
    },
  },
  plugins: [],
}
