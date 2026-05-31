import { create } from 'zustand'
import type { Emotion, EmotionResult, Voice, ProsodyParams } from '../types'

interface AppStore {
  // Voices
  voices: Voice[]
  selectedVoice: string
  setVoices: (voices: Voice[]) => void
  setSelectedVoice: (id: string) => void

  // Emotion
  selectedEmotion: Emotion | null
  currentEmotion: EmotionResult | null
  setSelectedEmotion: (e: Emotion | null) => void
  setCurrentEmotion: (e: EmotionResult | null) => void

  // Prosody overrides
  pitchOverride: number | null
  energyOverride: number | null
  speedOverride: number | null
  setPitchOverride: (v: number | null) => void
  setEnergyOverride: (v: number | null) => void
  setSpeedOverride: (v: number | null) => void

  // Language
  language: string
  setLanguage: (l: string) => void

  // Playback state
  isLoading: boolean
  isStreaming: boolean
  isPlaying: boolean
  latencyMs: number | null
  setIsLoading: (v: boolean) => void
  setIsStreaming: (v: boolean) => void
  setIsPlaying: (v: boolean) => void
  setLatencyMs: (ms: number | null) => void

  // Error
  error: string | null
  setError: (e: string | null) => void

  // Audio history
  history: Array<{
    id: string
    text: string
    emotion: string
    audioUrl: string
    duration: number
    createdAt: Date
  }>
  addToHistory: (item: AppStore['history'][0]) => void
  clearHistory: () => void
}

export const useAppStore = create<AppStore>((set) => ({
  voices: [],
  selectedVoice: 'default',
  setVoices: (voices) => set({ voices }),
  setSelectedVoice: (selectedVoice) => set({ selectedVoice }),

  selectedEmotion: null,
  currentEmotion: null,
  setSelectedEmotion: (selectedEmotion) => set({ selectedEmotion }),
  setCurrentEmotion: (currentEmotion) => set({ currentEmotion }),

  pitchOverride: null,
  energyOverride: null,
  speedOverride: null,
  setPitchOverride: (pitchOverride) => set({ pitchOverride }),
  setEnergyOverride: (energyOverride) => set({ energyOverride }),
  setSpeedOverride: (speedOverride) => set({ speedOverride }),

  language: 'en',
  setLanguage: (language) => set({ language }),

  isLoading: false,
  isStreaming: false,
  isPlaying: false,
  latencyMs: null,
  setIsLoading: (isLoading) => set({ isLoading }),
  setIsStreaming: (isStreaming) => set({ isStreaming }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setLatencyMs: (latencyMs) => set({ latencyMs }),

  error: null,
  setError: (error) => set({ error }),

  history: [],
  addToHistory: (item) => set((state) => ({
    history: [item, ...state.history].slice(0, 20),
  })),
  clearHistory: () => set({ history: [] }),
}))
