import './App.css'
import { useAuth } from './context/AuthContext'
import AuthScreen from './components/AuthScreen'
import Dashboard from './components/Dashboard'

// Top-level gate: decide between the loading splash, the auth screen, and the
// authenticated dashboard based on the auth status.
export default function App() {
  const { status } = useAuth()

  if (status === 'loading') {
    return (
      <div className="app-splash">
        <div className="bg-glow" aria-hidden="true" />
        <span className="spinner spinner-lg" aria-hidden="true" />
        <p>Loading…</p>
      </div>
    )
  }

  if (status === 'unauthenticated') {
    return <AuthScreen />
  }

  return <Dashboard />
}
