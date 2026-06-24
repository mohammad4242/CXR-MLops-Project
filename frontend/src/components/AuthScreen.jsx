import { useState } from 'react'
import { useAuth } from '../context/AuthContext'

// Sign in / sign up screen shown when no user is authenticated.
export default function AuthScreen() {
  const { login, register } = useAuth()

  const [mode, setMode] = useState('login') // 'login' | 'register'
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const isRegister = mode === 'register'

  const switchMode = (next) => {
    setMode(next)
    setError('')
    setPassword('')
    setConfirm('')
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')

    if (isRegister && password !== confirm) {
      setError('Passwords do not match.')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }

    setLoading(true)
    try {
      if (isRegister) {
        await register(email, password, fullName)
      } else {
        await login(email, password)
      }
      // On success the AuthProvider flips status and this screen unmounts.
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-screen">
      <div className="bg-glow" aria-hidden="true" />
      <div className="bg-grid" aria-hidden="true" />

      <div className="auth-card">
        <div className="auth-brand">
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

        <h1 className="auth-title">{isRegister ? 'Create your account' : 'Welcome back'}</h1>
        <p className="auth-sub">
          {isRegister
            ? 'Sign up to start analyzing chest X-rays.'
            : 'Sign in to continue to your dashboard.'}
        </p>

        <div className="auth-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={!isRegister}
            className={!isRegister ? 'active' : ''}
            onClick={() => switchMode('login')}
          >
            Sign in
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={isRegister}
            className={isRegister ? 'active' : ''}
            onClick={() => switchMode('register')}
          >
            Sign up
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {isRegister && (
            <label className="field">
              <span>Full name <em>(optional)</em></span>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Jane Doe"
                autoComplete="name"
              />
            </label>
          )}

          <label className="field">
            <span>Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
              autoComplete={isRegister ? 'new-password' : 'current-password'}
            />
          </label>

          {isRegister && (
            <label className="field">
              <span>Confirm password</span>
              <input
                type="password"
                required
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="Re-enter your password"
                autoComplete="new-password"
              />
            </label>
          )}

          {error && (
            <div className="banner banner-error" role="alert">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          <button className="cta" type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Please wait…
              </>
            ) : isRegister ? (
              'Create account'
            ) : (
              'Sign in'
            )}
          </button>
        </form>

        <p className="auth-switch">
          {isRegister ? 'Already have an account? ' : "Don't have an account? "}
          <button
            type="button"
            className="link-btn"
            onClick={() => switchMode(isRegister ? 'login' : 'register')}
          >
            {isRegister ? 'Sign in' : 'Sign up'}
          </button>
        </p>

        <p className="auth-disclaimer">
          For research and educational use only — not a medical device.
        </p>
      </div>
    </div>
  )
}
