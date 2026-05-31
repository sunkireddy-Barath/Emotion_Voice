import { useState } from 'react'
import { Brain, RefreshCcw } from 'lucide-react'
import { analyzeEmotion } from '../api/client'
import { EmotionBadge } from '../components/ui/EmotionBadge'
import type { EmotionResult } from '../types'

export default function EmotionPage() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<EmotionResult & { sentences?: EmotionResult[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [perSentence, setPerSentence] = useState(false)
  const [language, setLanguage] = useState('en')

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await analyzeEmotion(text, language)
      setResult(res as EmotionResult & { sentences?: EmotionResult[] })
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Brain className="text-indigo-400" size={28} />
          Emotion Analysis
        </h1>
        <p className="text-slate-400 mt-1">Detect emotion and intensity from text</p>
      </div>

      <div className="card space-y-4">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          className="input resize-none"
          placeholder="Enter text to analyze..."
        />

        <div className="flex items-center gap-4 flex-wrap">
          <select value={language} onChange={(e) => setLanguage(e.target.value)} className="select w-32">
            {['en', 'ta', 'hi', 'fr', 'de', 'es'].map((l) => (
              <option key={l} value={l}>{l.toUpperCase()}</option>
            ))}
          </select>

          <label className="flex items-center gap-2 text-slate-400 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={perSentence}
              onChange={(e) => setPerSentence(e.target.checked)}
              className="accent-indigo-500"
            />
            Per-sentence analysis
          </label>

          <button onClick={analyze} disabled={loading || !text.trim()} className="btn-primary ml-auto">
            {loading ? <RefreshCcw size={16} className="animate-spin" /> : <Brain size={16} />}
            Analyze
          </button>
        </div>
      </div>

      {result && (
        <div className="card space-y-6">
          <div className="flex items-center gap-4">
            <EmotionBadge emotion={result.emotion} size="lg" />
            <div>
              <p className="text-slate-400 text-sm">Intensity</p>
              <div className="flex items-center gap-2">
                <div className="w-40 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 rounded-full transition-all"
                    style={{ width: `${result.intensity * 100}%` }}
                  />
                </div>
                <span className="text-white font-mono text-sm">{(result.intensity * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Score bars */}
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-3">All Emotion Scores</h3>
            <div className="space-y-2">
              {Object.entries(result.scores)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, score]) => (
                  <div key={emotion} className="flex items-center gap-3">
                    <EmotionBadge emotion={emotion} size="sm" />
                    <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className="text-slate-400 font-mono text-xs w-12 text-right">
                      {(score * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
