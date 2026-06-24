import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import * as api from '../lib/api'

const AuthContext = createContext(null)

/**
 * Holds the authentication state for the whole app:
 *   status: 'loading' | 'authenticated' | 'unauthenticated'
 * On mount it validates any stored token by calling /auth/me.
 */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [status, setStatus] = useState('loading')

  useEffect(() => {
    let active = true
    const token = api.getToken()

    if (!token) {
      setStatus('unauthenticated')
      return
    }

    api
      .getMe()
      .then((me) => {
        if (!active) return
        setUser(me)
        setStatus('authenticated')
      })
      .catch(() => {
        if (!active) return
        api.clearToken()
        setUser(null)
        setStatus('unauthenticated')
      })

    return () => {
      active = false
    }
  }, [])

  const login = useCallback(async (email, password) => {
    const data = await api.login({ email, password })
    api.setToken(data.access_token)
    setUser(data.user)
    setStatus('authenticated')
    return data.user
  }, [])

  const register = useCallback(
    async (email, password, fullName) => {
      await api.register({ email, password, fullName })
      // Seamlessly sign the new user in.
      return login(email, password)
    },
    [login],
  )

  const logout = useCallback(() => {
    api.clearToken()
    setUser(null)
    setStatus('unauthenticated')
  }, [])

  const value = { user, status, login, register, logout }
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
