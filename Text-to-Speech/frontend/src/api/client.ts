/**
 * API client for the Emotion Voice backend.
 *
 * In development: requests go to /api/* → Vite proxy → localhost:8000
 * In production:  requests go to /api/* → nginx → api container
 *
 * API key is read from VITE_API_KEY env var (set in .env.local).
 */
import axios from 'axios'
import type { EmotionResult, Voice, TTSRequest } from '../types'

const BASE = '/api'
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined

const http = axios.create({
  baseURL: BASE,
  timeout: 120_000,
  headers: {
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
})

// ─── TTS ──────────────────────────────────────────────────────────────────────

export async function synthesize(
  req: TTSRequest,
): Promise<{ blob: Blob; emotion: string; latencyMs: number; durationSec: number; model: string }> {
  const response = await http.post('/tts', req, { responseType: 'blob' })
  return {
    blob: response.data as Blob,
    emotion: response.headers['x-emotion'] || 'neutral',
    latencyMs: parseFloat(response.headers['x-latency-ms'] || '0'),
    durationSec: parseFloat(response.headers['x-duration-sec'] || '0'),
    model: response.headers['x-model'] || 'unknown',
  }
}

export async function synthesizeJson(req: TTSRequest): Promise<{
  audio_base64: string
  emotion: string
  intensity: number
  prosody: { pitch: number; energy: number; speed: number; pause_factor: number }
  duration_sec: number
  latency_ms: number
  model_used: string
  request_id: string
}> {
  const { data } = await http.post('/tts/json', req)
  return data
}

// ─── Streaming ────────────────────────────────────────────────────────────────

export async function* streamSSE(req: {
  text: string
  voice_id?: string
  language?: string
  emotion?: string | null
  speed?: number | null
  sample_rate?: number
}): AsyncGenerator<{ audio_base64: string; chunk_id: number; is_last: boolean; emotion: string }> {
  const url = `/api/stream`
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
    },
    body: JSON.stringify(req),
  })

  if (!resp.ok) throw new Error(`Stream failed: ${resp.status}`)
  if (!resp.body) throw new Error('No response body')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (!data || data === '{"type": "done"}') continue
        try {
          yield JSON.parse(data)
        } catch {
          // skip malformed chunk
        }
      }
    }
  }
}

/**
 * Create a WebSocket stream connection.
 * Uses the Vite proxy path /ws/stream → ws://localhost:8000/ws/stream
 */
export function createWebSocketStream(
  onChunk: (audioData: ArrayBuffer, meta: Record<string, unknown>) => void,
  onEnd: () => void,
  onError: (err: string) => void,
): { send: (payload: object) => void; close: () => void } {
  // Relative ws URL — works through nginx reverse proxy and Vite proxy
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`
  const ws = new WebSocket(wsUrl)

  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    if (API_KEY) {
      ws.send(JSON.stringify({ type: 'auth', api_key: API_KEY }))
    }
  }

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      try {
        const msg = JSON.parse(event.data) as { type: string; message?: string }
        if (msg.type === 'stream_end') { onEnd(); return }
        if (msg.type === 'error') { onError(msg.message ?? 'Unknown error'); return }
      } catch { /* ignore */ }
      return
    }
    // Binary frame: [4-byte BE meta_len][JSON meta][WAV bytes]
    const buf = event.data as ArrayBuffer
    const view = new DataView(buf)
    const metaLen = view.getUint32(0)
    const metaBytes = new Uint8Array(buf, 4, metaLen)
    const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as Record<string, unknown>
    const audioData = buf.slice(4 + metaLen)
    onChunk(audioData, meta)
  }

  ws.onerror = () => onError('WebSocket connection error')
  ws.onclose = (e) => { if (!e.wasClean) onError('WebSocket closed unexpectedly') }

  return {
    send: (payload) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload))
    },
    close: () => ws.close(1000, 'Client closed'),
  }
}

// ─── Emotion ──────────────────────────────────────────────────────────────────

export async function analyzeEmotion(
  text: string,
  language = 'en',
  perSentence = false,
): Promise<EmotionResult & { sentences?: EmotionResult[] }> {
  const { data } = await http.post('/emotion-analysis', { text, language, per_sentence: perSentence })
  return data
}

// ─── Voices ───────────────────────────────────────────────────────────────────

export async function listVoices(): Promise<Voice[]> {
  const { data } = await http.get('/voices')
  return (data as { voices: Voice[] }).voices
}

export async function createVoice(name: string, language = 'en', description = ''): Promise<Voice> {
  const { data } = await http.post('/voices', { name, language, description })
  return data as Voice
}

export async function uploadVoiceSample(
  voiceId: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<{ filename: string; duration: number; total_samples: number }> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await http.post(`/voices/${voiceId}/samples`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

export async function deleteVoice(voiceId: string): Promise<void> {
  await http.delete(`/voices/${voiceId}`)
}

export async function validateVoice(voiceId: string): Promise<{
  valid: boolean; issues: string[]; sample_count: number; total_duration: number
}> {
  const { data } = await http.get(`/voices/${voiceId}/validate`)
  return data
}

// ─── System ───────────────────────────────────────────────────────────────────

export async function health(): Promise<{
  status: string; model_type: string; ready: boolean
}> {
  const { data } = await http.get('/health')
  return data
}

export async function systemInfo() {
  const { data } = await http.get('/info')
  return data
}
