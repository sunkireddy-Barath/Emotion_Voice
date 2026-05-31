export type Emotion =
  | 'neutral' | 'happy' | 'sad' | 'angry' | 'excited'
  | 'fear' | 'surprise' | 'calm' | 'serious'
  | 'motivational' | 'questioning' | 'storytelling'

export interface EmotionResult {
  emotion: Emotion
  intensity: number
  confidence: number
  scores: Record<string, number>
}

export interface ProsodyParams {
  pitch: number
  energy: number
  speed: number
  pause_factor: number
}

export interface Voice {
  voice_id: string
  name: string
  language: string
  description: string
  sample_count: number
  total_duration_sec: number
  created_at: string
  type: 'built-in' | 'cloned'
}

export interface TTSRequest {
  text: string
  voice_id?: string
  language?: string
  emotion?: Emotion | null
  intensity?: number | null
  pitch?: number | null
  energy?: number | null
  speed?: number | null
  sample_rate?: number
}

export interface TTSResult {
  audioBlob: Blob
  emotion: EmotionResult
  prosody: ProsodyParams
  duration_sec: number
  latency_ms: number
  model_used: string
}

export interface StreamChunk {
  chunk_id: number
  audio_base64: string
  duration_sec: number
  emotion: string
  is_last: boolean
  latency_ms: number
}

export interface AppState {
  isLoading: boolean
  error: string | null
  voices: Voice[]
  selectedVoice: string
  selectedEmotion: Emotion | null
  currentEmotion: EmotionResult | null
  isStreaming: boolean
  isPlaying: boolean
}
