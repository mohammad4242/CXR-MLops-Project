// Live backend status pill shown in the top bar. Polls /health from App and
// reflects three states: checking, online, offline. Clicking retries.
export default function HealthBadge({ status, health, onRefresh }) {
  const label =
    status === 'online' ? 'API online' : status === 'offline' ? 'API offline' : 'Checking…'

  const title = health
    ? `Model: ${health.model_name}\nVersion: ${health.version}\nUptime: ${Math.round(
        health.uptime_seconds,
      )}s`
    : 'Click to retry the connection'

  return (
    <button
      className={`health health-${status}`}
      type="button"
      onClick={onRefresh}
      title={title}
    >
      <span className="health-dot" aria-hidden="true" />
      <span className="health-label">{label}</span>
      {status === 'online' && health?.model_name && (
        <span className="health-model">{health.model_name}</span>
      )}
    </button>
  )
}
