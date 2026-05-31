import { useState, useCallback, useRef } from 'react'
import { Mic2, Zap, RefreshCcw, Settings2 } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '../store/appStore'
import { synthesize, analyzeEmotion } from '../api/client'
import { EmotionBadge, EmotionSelector } from '../components/ui/EmotionBadge'
import { WaveformVisualizer, WaveformAnimation } from '../components/waveform/WaveformVisualizer'
import type { Emotion, EmotionResult } from '../types'

const LANGUAGES = [
  { code: 'en', label: 'English', voice: 'en-US Neural (Christopher / Emma / Aria...)' },
  { code: 'ta', label: 'Tamil', voice: 'ta-IN Neural (Valluvar / Pallavi)' },
  { code: 'hi', label: 'Hindi', voice: 'hi-IN Neural (Madhur)' },
  { code: 'fr', label: 'French', voice: 'fr-FR Neural (Henri)' },
  { code: 'de', label: 'German', voice: 'de-DE Neural (Conrad)' },
  { code: 'es', label: 'Spanish', voice: 'es-ES Neural (Alvaro)' },
]

export default function SynthesizePage() {
  const {
    voices, selectedVoice, setSelectedVoice,
    selectedEmotion, setSelectedEmotion,
    currentEmotion, setCurrentEmotion,
    language, setLanguage,
    pitchOverride, setPitchOverride,
    energyOverride, setEnergyOverride,
    speedOverride, setSpeedOverride,
    isLoading, setIsLoading,
    error, setError,
    addToHistory,
  } = useAppStore()

  const [text, setText] = useState('')
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [showProsody, setShowProsody] = useState(false)
  const [useStreaming, setUseStreaming] = useState(false)
  const audioQueueRef = useRef<ArrayBuffer[]>([])

  const handleSynthesize = useCallback(async () => {
    if (!text.trim()) return
    setIsLoading(true)
    setError(null)
    setAudioBlob(null)

    const t0 = Date.now()
    try {
      const result = await synthesize({
        text,
        voice_id: selectedVoice,
        language,
        emotion: selectedEmotion,
        pitch: pitchOverride,
        energy: energyOverride,
        speed: speedOverride,
      })

      setAudioBlob(result.blob)
      setLatencyMs(result.latencyMs || Date.now() - t0)

      if (result.emotion) {
        setCurrentEmotion({ emotion: result.emotion as Emotion, intensity: 0.8, confidence: 0.8, scores: {} })
      }

      addToHistory({
        id: Date.now().toString(),
        text: text.slice(0, 80),
        emotion: result.emotion || 'neutral',
        audioUrl: URL.createObjectURL(result.blob),
        duration: result.durationSec || 0,
        createdAt: new Date(),
      })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Synthesis failed')
    } finally {
      setIsLoading(false)
    }
  }, [text, selectedVoice, language, selectedEmotion, pitchOverride, energyOverride, speedOverride])

  const handleDetectEmotion = async () => {
    if (!text.trim()) return
    try {
      const result = await analyzeEmotion(text, language)
      setCurrentEmotion(result)
    } catch (err) {
      console.error('Emotion detection failed:', err)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Mic2 className="text-indigo-400" size={28} />
          Speech Synthesis
        </h1>
        <p className="text-slate-400 mt-1">Convert text to emotion-aware speech</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Text + controls */}
        <div className="lg:col-span-2 space-y-4">
          {/* Text input */}
          <div className="card space-y-3">
            <label className="label">Input Text</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste your text here..."
              rows={6}
              className="input resize-none"
            />
            <div className="flex items-center justify-between text-sm text-slate-500">
              <span>{text.length} / 5000 chars</span>
              <button
                onClick={handleDetectEmotion}
                disabled={!text.trim()}
                className="text-indigo-400 hover:text-indigo-300 disabled:opacity-40 flex items-center gap-1"
              >
                <RefreshCcw size={13} /> Detect Emotion
              </button>
            </div>
          </div>

          {/* Emotion selector */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <label className="label mb-0">Emotion</label>
              {currentEmotion && (
                <EmotionBadge emotion={currentEmotion.emotion} intensity={currentEmotion.intensity} showIntensity />
              )}
            </div>
            <EmotionSelector value={selectedEmotion} onChange={(e) => setSelectedEmotion(e as Emotion | null)} />
          </div>

          {/* Audio output */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <label className="label mb-0">Audio Output</label>
              {latencyMs && (
                <span className="text-xs text-slate-500">
                  Generated in {latencyMs}ms
                </span>
              )}
            </div>
            {isLoading ? (
              <div className="flex items-center gap-3 py-4">
                <WaveformAnimation active />
                <span className="text-slate-400 text-sm">Generating speech...</span>
              </div>
            ) : (
              <WaveformVisualizer audioBlob={audioBlob} />
            )}
          </div>

          {error && (
            <div className="bg-red-950/50 border border-red-700 text-red-300 rounded-xl p-4 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right: Settings */}
        <div className="space-y-4">
          {/* Voice selector */}
          <div className="card space-y-3">
            <label className="label">Voice</label>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="select"
            >
              <option value="default">Default Voice</option>
              {voices.map((v) => (
                <option key={v.voice_id} value={v.voice_id}>
                  {v.name} ({v.language.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          {/* Language */}
          <div className="card space-y-3">
            <label className="label">Language</label>
            <select value={language} onChange={(e) => setLanguage(e.target.value)} className="select">
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>{l.label}</option>
              ))}
            </select>
            <p className="text-xs text-slate-500">
              Voice: {LANGUAGES.find(l => l.code === language)?.voice ?? 'Neural'}
            </p>
          </div>

          {/* Prosody controls */}
          <div className="card space-y-3">
            <button
              onClick={() => setShowProsody(!showProsody)}
              className="flex items-center gap-2 text-slate-300 hover:text-white w-full"
            >
              <Settings2 size={16} className="text-indigo-400" />
              <span className="label mb-0 cursor-pointer">Prosody Override</span>
            </button>

            {showProsody && (
              <div className="space-y-4 pt-2 border-t border-slate-700">
                {[
                  { label: 'Pitch', value: pitchOverride, set: setPitchOverride },
                  { label: 'Energy', value: energyOverride, set: setEnergyOverride },
                  { label: 'Speed',  value: speedOverride,  set: setSpeedOverride },
                ].map(({ label, value, set }) => (
                  <div key={label}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400">{label}</span>
                      <span className="text-slate-300 font-mono">{value?.toFixed(2) ?? 'Auto'}</span>
                    </div>
                    <input
                      type="range"
                      min={0.5} max={2.0} step={0.05}
                      value={value ?? 1.0}
                      onChange={(e) => set(parseFloat(e.target.value))}
                      className="w-full accent-indigo-500"
                    />
                    <button
                      onClick={() => set(null)}
                      className="text-xs text-slate-500 hover:text-slate-400 mt-1"
                    >
                      Reset to auto
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Synthesize button */}
          <button
            onClick={handleSynthesize}
            disabled={isLoading || !text.trim()}
            className="btn-primary w-full py-3 text-base justify-center"
          >
            {isLoading ? (
              <>
                <RefreshCcw size={18} className="animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Zap size={18} />
                Synthesize Speech
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
