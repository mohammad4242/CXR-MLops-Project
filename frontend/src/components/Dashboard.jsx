import { useCallback, useEffect, useRef, useState } from 'react'
import Dropzone from './Dropzone'
import HealthBadge from './HealthBadge'
import Results from './Results'
import { useAuth } from '../context/AuthContext'
import { checkHealth, predict } from '../lib/api'

// The authenticated experience: upload an X-ray, run inference, view findings.
export default function Dashboard() {
  const { user, logout } = useAuth()

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [health, setHealth] = useState(null)
  const [healthStatus, setHealthStatus] = useState('checking') // checking | online | offline

  const previewRef = useRef('')

  const refreshHealth = useCallback(async () => {
    setHealthStatus((current) => (current === 'online' ? current : 'checking'))
    try {
      const data = await checkHealth()
      setHealth(data)
      setHealthStatus('online')
    } catch {
      setHealth(null)
      setHealthStatus('offline')
    }
  }, [])

  useEffect(() => {
    refreshHealth()
    const interval = setInterval(refreshHealth, 30000)
    return () => clearInterval(interval)
  }, [refreshHealth])

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    },
    [],
  )

  const selectFile = useCallback((nextFile) => {
    setError('')
    setResult(null)

    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current)
      previewRef.current = ''
    }

    if (!nextFile) {
      setFile(null)
      setPreviewUrl('')
      return
    }

    if (!nextFile.type.startsWith('image/')) {
      setError('Please choose an image file (PNG or JPEG).')
      setFile(null)
      setPreviewUrl('')
      return
    }

    const url = URL.createObjectURL(nextFile)
    previewRef.current = url
    setFile(nextFile)
    setPreviewUrl(url)
  }, [])

  const clearFile = useCallback(() => selectFile(null), [selectFile])

  const runPrediction = useCallback(async () => {
    if (!file) {
      setError('Please choose an X-ray image first.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await predict(file)
      setResult(data)
    } catch (err) {
      // If the token expired or is invalid, send the user back to sign in.
      if (err.status === 401) {
        logout()
        return
      }
      setError(err.message || 'Something went wrong while analyzing the image.')
    } finally {
      setLoading(false)
    }
  }, [file, logout])

  const displayName = user?.full_name || user?.email || 'Account'
  const initial = displayName.charAt(0).toUpperCase()

  return (
    <div className="app">
      <div className="bg-glow" aria-hidden="true" />
      <div className="bg-grid" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12h3l2-5 4 10 2-5h3" />
            </svg>
          </span>
          <span className="brand-text">
            <strong>CXR Insight</strong>
            <small>Chest X-ray Analysis</small>
          </span>
        </div>

        <div className="topbar-right">
          <HealthBadge status={healthStatus} health={health} onRefresh={refreshHealth} />
          <div className="user-menu">
            <span className="avatar" aria-hidden="true">{initial}</span>
            <span className="user-name" title={user?.email}>{displayName}</span>
            <button className="logout-btn" type="button" onClick={logout} title="Sign out">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <path d="m16 17 5-5-5-5" />
                <path d="M21 12H9" />
              </svg>
              <span>Sign out</span>
            </button>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="hero">
          <p className="eyebrow">AI-assisted radiology</p>
          <h1>Analyze a chest X-ray in seconds</h1>
          <p className="lede">
            Upload a chest radiograph and a DenseNet-121 model from TorchXRayVision estimates
            the likelihood of thoracic findings. Only results above 65% confidence are reported.
          </p>
        </section>

        <div className="grid">
          <Dropzone
            file={file}
            previewUrl={previewUrl}
            loading={loading}
            onSelect={selectFile}
            onClear={clearFile}
            onAnalyze={runPrediction}
          />

          <div className="result-col">
            {error && (
              <div className="banner banner-error" role="alert">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="9" />
                  <path d="M12 8v4M12 16h.01" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            {loading && <AnalyzingState />}

            {!loading && result && <Results key={result.id ?? 'result'} result={result} />}

            {!loading && !result && !error && <EmptyState />}
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>
          For research and educational use only — this is not a medical device and not a
          substitute for diagnosis by a qualified clinician.
        </p>
      </footer>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="placeholder">
      <span className="placeholder-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="3" />
          <path d="M3 14l4-4 4 4 3-3 7 7" />
          <circle cx="8.5" cy="8.5" r="1.5" />
        </svg>
      </span>
      <h3>No analysis yet</h3>
      <p>Upload a chest X-ray and select “Analyze” to see ranked predictions here.</p>
    </div>
  )
}

function AnalyzingState() {
  return (
    <div className="placeholder analyzing">
      <span className="pulse-ring" aria-hidden="true" />
      <h3>Analyzing the radiograph…</h3>
      <p>Running inference through the DenseNet-121 model. This usually takes a moment.</p>
    </div>
  )
}
