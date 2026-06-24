// Thin client around the FastAPI backend.
//
// Every request goes to a relative "/api" path. In development Vite proxies it
// to http://localhost:8000; in production Nginx proxies it to the "api"
// container. Because the path is relative, the browser treats it as same-origin
// and CORS never comes into play.
const API_BASE = '/api'
const TOKEN_KEY = 'cxr_token'


// ---- Token storage ---------------------------------------------------------

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}


// ---- Error helper ----------------------------------------------------------

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function readJson(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function messageFrom(data, fallback) {
  const detail = data?.detail
  if (typeof detail === 'string') return detail
  // FastAPI validation errors come back as a list of {msg, loc} objects.
  if (Array.isArray(detail) && detail.length) {
    return detail[0]?.msg || fallback
  }
  return fallback
}


// ---- Endpoints -------------------------------------------------------------

/** GET /health — backend status, model name, uptime. */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new ApiError(`Health check failed (${response.status})`, response.status)
  }
  return response.json()
}

/** POST /auth/register — create a new account. */
export async function register({ email, password, fullName }) {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, full_name: fullName || null }),
  })
  const data = await readJson(response)
  if (!response.ok) {
    throw new ApiError(messageFrom(data, 'Registration failed.'), response.status)
  }
  return data
}

/** POST /auth/login — exchange credentials for an access token. */
export async function login({ email, password }) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  const data = await readJson(response)
  if (!response.ok) {
    throw new ApiError(messageFrom(data, 'Login failed.'), response.status)
  }
  return data // { access_token, token_type, user }
}

/** GET /auth/me — the currently authenticated user (uses the stored token). */
export async function getMe() {
  const response = await fetch(`${API_BASE}/auth/me`, { headers: { ...authHeaders() } })
  const data = await readJson(response)
  if (!response.ok) {
    throw new ApiError(messageFrom(data, 'Not authenticated.'), response.status)
  }
  return data
}

/**
 * POST /predict — upload one chest X-ray image (requires authentication).
 * Returns only findings at/above the backend's confidence threshold.
 */
export async function predict(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    // Note: never set Content-Type for FormData — the browser adds the boundary.
    headers: { ...authHeaders() },
    body: formData,
  })

  const data = await readJson(response)
  if (!response.ok) {
    throw new ApiError(
      messageFrom(data, `Prediction failed (${response.status})`),
      response.status,
    )
  }
  return data
}
