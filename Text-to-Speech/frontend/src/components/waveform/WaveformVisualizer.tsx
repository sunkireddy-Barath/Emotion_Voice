import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, Square, Download } from 'lucide-react'

interface Props {
  audioBlob?: Blob | null
  audioUrl?: string | null
  onPlayStateChange?: (playing: boolean) => void
}

export function WaveformVisualizer({ audioBlob, audioUrl, onPlayStateChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    if (!containerRef.current) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4c6ef5',
      progressColor: '#748ffc',
      cursorColor: '#5c7cfa',
      height: 80,
      normalize: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
    })

    ws.on('play',   () => { setIsPlaying(true);  onPlayStateChange?.(true) })
    ws.on('pause',  () => { setIsPlaying(false); onPlayStateChange?.(false) })
    ws.on('finish', () => { setIsPlaying(false); onPlayStateChange?.(false) })
    ws.on('ready',  () => setDuration(ws.getDuration()))
    ws.on('audioprocess', () => setCurrentTime(ws.getCurrentTime()))

    wavesurferRef.current = ws

    return () => ws.destroy()
  }, [])

  useEffect(() => {
    if (!wavesurferRef.current) return
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob)
      wavesurferRef.current.load(url)
      return () => URL.revokeObjectURL(url)
    }
    if (audioUrl) {
      wavesurferRef.current.load(audioUrl)
    }
  }, [audioBlob, audioUrl])

  const togglePlay = () => wavesurferRef.current?.playPause()
  const stop = () => { wavesurferRef.current?.stop(); setCurrentTime(0) }

  const download = () => {
    if (!audioBlob) return
    const url = URL.createObjectURL(audioBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'speech.wav'
    a.click()
    URL.revokeObjectURL(url)
  }

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  const hasAudio = !!(audioBlob || audioUrl)

  return (
    <div className="space-y-3">
      {/* Waveform container */}
      <div
        ref={containerRef}
        className="bg-slate-900 rounded-xl border border-slate-700 p-3 min-h-[100px]"
      />

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            disabled={!hasAudio}
            className="btn-primary p-2 rounded-full disabled:opacity-40"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            onClick={stop}
            disabled={!hasAudio}
            className="btn-secondary p-2 rounded-full disabled:opacity-40"
            aria-label="Stop"
          >
            <Square size={16} />
          </button>
        </div>

        <span className="text-sm text-slate-400 font-mono">
          {fmt(currentTime)} / {fmt(duration)}
        </span>

        <button
          onClick={download}
          disabled={!audioBlob}
          className="btn-secondary py-1.5 text-sm disabled:opacity-40"
          aria-label="Download WAV"
        >
          <Download size={15} /> Download
        </button>
      </div>
    </div>
  )
}

/* Animated bars shown while generating */
export function WaveformAnimation({ active = true }: { active?: boolean }) {
  if (!active) return null
  return (
    <div className="flex items-end gap-0.5 h-10">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="waveform-bar h-4" style={{ animationDelay: `${i * 0.1}s` }} />
      ))}
    </div>
  )
}
