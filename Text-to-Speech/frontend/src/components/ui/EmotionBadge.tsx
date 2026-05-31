import { clsx } from 'clsx'
import type { Emotion } from '../../types'

const EMOTION_CONFIG: Record<string, { label: string; color: string; emoji: string }> = {
  neutral:      { label: 'Neutral',      color: 'bg-gray-600 text-gray-100',    emoji: '😐' },
  happy:        { label: 'Happy',        color: 'bg-yellow-500 text-yellow-950', emoji: '😊' },
  sad:          { label: 'Sad',          color: 'bg-blue-600 text-blue-100',     emoji: '😢' },
  angry:        { label: 'Angry',        color: 'bg-red-600 text-red-100',       emoji: '😡' },
  excited:      { label: 'Excited',      color: 'bg-orange-500 text-orange-950', emoji: '🤩' },
  fear:         { label: 'Fear',         color: 'bg-purple-600 text-purple-100', emoji: '😨' },
  surprise:     { label: 'Surprise',     color: 'bg-pink-500 text-pink-950',     emoji: '😲' },
  calm:         { label: 'Calm',         color: 'bg-emerald-600 text-emerald-100', emoji: '😌' },
  serious:      { label: 'Serious',      color: 'bg-slate-600 text-slate-100',   emoji: '🎯' },
  motivational: { label: 'Motivational', color: 'bg-cyan-500 text-cyan-950',     emoji: '💪' },
  questioning:  { label: 'Questioning',  color: 'bg-violet-500 text-violet-950', emoji: '🤔' },
  storytelling: { label: 'Storytelling', color: 'bg-lime-500 text-lime-950',     emoji: '📖' },
}

interface Props {
  emotion: string
  intensity?: number
  size?: 'sm' | 'md' | 'lg'
  showIntensity?: boolean
}

export function EmotionBadge({ emotion, intensity, size = 'md', showIntensity = false }: Props) {
  const config = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral

  const sizeClass = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5',
  }[size]

  return (
    <span className={clsx('badge font-medium', config.color, sizeClass)}>
      {config.emoji} {config.label}
      {showIntensity && intensity !== undefined && (
        <span className="ml-1.5 opacity-80">({Math.round(intensity * 100)}%)</span>
      )}
    </span>
  )
}

export function EmotionSelector({
  value,
  onChange,
  includeAuto = true,
}: {
  value: string | null
  onChange: (e: string | null) => void
  includeAuto?: boolean
}) {
  const emotions = Object.entries(EMOTION_CONFIG)

  return (
    <div className="grid grid-cols-4 gap-2">
      {includeAuto && (
        <button
          onClick={() => onChange(null)}
          className={clsx(
            'px-3 py-2 rounded-lg text-sm font-medium border transition-all',
            value === null
              ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
              : 'border-slate-600 text-slate-400 hover:border-slate-500'
          )}
        >
          🤖 Auto
        </button>
      )}
      {emotions.map(([key, cfg]) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={clsx(
            'px-2 py-2 rounded-lg text-xs font-medium border transition-all',
            value === key
              ? 'border-indigo-500 bg-indigo-600/20 text-indigo-300'
              : 'border-slate-600 text-slate-400 hover:border-slate-500'
          )}
        >
          {cfg.emoji} {cfg.label}
        </button>
      ))}
    </div>
  )
}
