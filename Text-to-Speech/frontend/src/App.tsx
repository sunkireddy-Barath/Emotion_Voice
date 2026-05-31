import { Routes, Route, NavLink, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { Mic2, UserCircle2, Brain, Activity, Waves } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from './store/appStore'
import { listVoices, health } from './api/client'
import SynthesizePage from './pages/SynthesizePage'
import VoicesPage from './pages/VoicesPage'
import EmotionPage from './pages/EmotionPage'

const NAV_ITEMS = [
  { to: '/',        icon: Mic2,        label: 'Synthesize' },
  { to: '/voices',  icon: UserCircle2, label: 'Voices' },
  { to: '/emotion', icon: Brain,       label: 'Emotion' },
]

export default function App() {
  const { setVoices, error } = useAppStore()

  useEffect(() => {
    // Load voices and check health on mount
    listVoices().then(setVoices).catch(console.error)
    health().catch(console.error)
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top nav */}
      <header className="bg-slate-800/80 backdrop-blur border-b border-slate-700 sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Waves size={18} className="text-white" />
            </div>
            <span className="font-bold text-white">Emotion Voice AI</span>
          </div>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-indigo-600/20 text-indigo-300'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                  )
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        <Routes>
          <Route path="/"        element={<SynthesizePage />} />
          <Route path="/voices"  element={<VoicesPage />} />
          <Route path="/emotion" element={<EmotionPage />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-4 text-center text-slate-600 text-sm">
        Emotion-Aware Speech Foundation Model · Self-hosted · No external APIs
      </footer>
    </div>
  )
}
