import { useState, useCallback, useEffect } from 'react'
import { UserCircle2, Plus, Trash2, Upload, CheckCircle2, AlertCircle } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { createVoice, deleteVoice, uploadVoiceSample, listVoices } from '../api/client'
import type { Voice } from '../types'

export default function VoicesPage() {
  const { voices, setVoices, setSelectedVoice } = useAppStore()
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newLang, setNewLang] = useState('en')
  const [uploading, setUploading] = useState<Record<string, boolean>>({})
  const [feedback, setFeedback] = useState<{ id: string; msg: string; ok: boolean } | null>(null)

  useEffect(() => {
    listVoices().then(setVoices).catch(console.error)
  }, [])

  const handleCreate = async () => {
    if (!newName.trim()) return
    try {
      const v = await createVoice(newName.trim(), newLang)
      setVoices([...voices, v])
      setNewName('')
      setCreating(false)
    } catch (err) {
      console.error(err)
    }
  }

  const handleDelete = async (voiceId: string) => {
    if (!confirm('Delete this voice profile? This cannot be undone.')) return
    try {
      await deleteVoice(voiceId)
      setVoices(voices.filter((v) => v.voice_id !== voiceId))
    } catch (err) {
      console.error(err)
    }
  }

  const handleUpload = async (voiceId: string, files: FileList | null) => {
    if (!files || files.length === 0) return
    setUploading((prev) => ({ ...prev, [voiceId]: true }))
    try {
      for (const file of Array.from(files)) {
        await uploadVoiceSample(voiceId, file)
      }
      const updated = await listVoices()
      setVoices(updated)
      setFeedback({ id: voiceId, msg: 'Sample uploaded successfully!', ok: true })
    } catch (err: unknown) {
      setFeedback({ id: voiceId, msg: String(err), ok: false })
    } finally {
      setUploading((prev) => ({ ...prev, [voiceId]: false }))
      setTimeout(() => setFeedback(null), 4000)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <UserCircle2 className="text-indigo-400" size={28} />
            Voice Profiles
          </h1>
          <p className="text-slate-400 mt-1">Train and manage cloned voice profiles</p>
        </div>
        <button onClick={() => setCreating(true)} className="btn-primary">
          <Plus size={16} /> New Voice
        </button>
      </div>

      {/* Create form */}
      {creating && (
        <div className="card space-y-4">
          <h2 className="font-semibold text-white">Create New Voice</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Voice Name</label>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. My Voice"
                className="input"
              />
            </div>
            <div>
              <label className="label">Language</label>
              <select value={newLang} onChange={(e) => setNewLang(e.target.value)} className="select">
                {['en', 'ta', 'hi', 'fr', 'de', 'es'].map((l) => (
                  <option key={l} value={l}>{l.toUpperCase()}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={handleCreate} className="btn-primary">Create</button>
            <button onClick={() => setCreating(false)} className="btn-secondary">Cancel</button>
          </div>
        </div>
      )}

      {/* Voice list */}
      {voices.length === 0 ? (
        <div className="card text-center py-16 text-slate-500">
          <UserCircle2 size={48} className="mx-auto mb-3 opacity-30" />
          <p>No voice profiles yet.</p>
          <p className="text-sm mt-1">Create one and upload audio samples to clone your voice.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {voices.map((voice) => (
            <VoiceCard
              key={voice.voice_id}
              voice={voice}
              uploading={!!uploading[voice.voice_id]}
              feedback={feedback?.id === voice.voice_id ? feedback : null}
              onDelete={() => handleDelete(voice.voice_id)}
              onUpload={(files) => handleUpload(voice.voice_id, files)}
              onSelect={() => setSelectedVoice(voice.voice_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function VoiceCard({
  voice, uploading, feedback, onDelete, onUpload, onSelect,
}: {
  voice: Voice
  uploading: boolean
  feedback: { msg: string; ok: boolean } | null
  onDelete: () => void
  onUpload: (files: FileList | null) => void
  onSelect: () => void
}) {
  const quality = voice.total_duration_sec >= 30 ? 'excellent'
    : voice.total_duration_sec >= 10 ? 'good'
    : voice.total_duration_sec >= 3  ? 'minimum'
    : 'insufficient'

  const qualityColor = {
    excellent: 'text-green-400',
    good: 'text-yellow-400',
    minimum: 'text-orange-400',
    insufficient: 'text-red-400',
  }[quality]

  return (
    <div className="card flex flex-col sm:flex-row gap-4">
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-white">{voice.name}</h3>
          <span className="badge bg-slate-700 text-slate-300 text-xs">{voice.language.toUpperCase()}</span>
        </div>
        <div className="text-sm text-slate-400 space-y-1">
          <p>{voice.sample_count} sample{voice.sample_count !== 1 ? 's' : ''}</p>
          <p className={qualityColor}>
            {voice.total_duration_sec.toFixed(1)}s recorded
            {' '}({quality} quality)
          </p>
        </div>
        {feedback && (
          <div className={`flex items-center gap-1.5 text-sm ${feedback.ok ? 'text-green-400' : 'text-red-400'}`}>
            {feedback.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
            {feedback.msg}
          </div>
        )}
      </div>

      <div className="flex flex-row sm:flex-col gap-2 justify-end">
        <label className="btn-secondary cursor-pointer text-sm">
          <Upload size={14} />
          {uploading ? 'Uploading...' : 'Add Sample'}
          <input
            type="file"
            accept=".wav,.mp3,.flac,.ogg"
            multiple
            className="hidden"
            onChange={(e) => onUpload(e.target.files)}
            disabled={uploading}
          />
        </label>
        <button onClick={onSelect} className="btn-primary text-sm">
          Use Voice
        </button>
        <button onClick={onDelete} className="btn-danger text-sm p-2" aria-label="Delete">
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  )
}
