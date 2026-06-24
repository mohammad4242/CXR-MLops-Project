import { useEffect, useMemo, useState } from 'react'
import { PATHOLOGY_INFO, prettyName, severity, SEVERITY_LABEL } from '../lib/pathologies'
import { formatDate, formatPercent } from '../lib/format'

// Clamp the rendered bar width: a 2% floor keeps borderline findings visible.
const barWidth = (score) => Math.max(2, Math.min(100, score * 100))

export default function Results({ result }) {
  const [animate, setAnimate] = useState(false)

  // The backend already filtered to significant findings; we just sort + decorate.
  const findings = useMemo(() => {
    const entries = Object.entries(result?.findings ?? {})
    return entries
      .sort((a, b) => b[1] - a[1])
      .map(([key, score]) => ({
        key,
        score,
        name: prettyName(key),
        desc: PATHOLOGY_INFO[key]?.desc ?? '',
        sev: severity(score),
      }))
  }, [result])

  // Replay the bar-fill animation whenever a fresh result arrives.
  useEffect(() => {
    setAnimate(false)
    const frame = requestAnimationFrame(() => setAnimate(true))
    return () => cancelAnimationFrame(frame)
  }, [result])

  const thresholdPct = Math.round((result?.threshold ?? 0.65) * 100)

  const meta = (
    <div className="meta-row">
      <span className="chip">
        <i>Record</i>#{result.id}
      </span>
      <span className="chip" title={result.filename}>
        <i>File</i>
        <span className="chip-value">{result.filename}</span>
      </span>
      <span className="chip">
        <i>Analyzed</i>
        {formatDate(result.saved_at)}
      </span>
    </div>
  )

  // ---- No significant findings ----
  if (!result.has_findings || findings.length === 0) {
    return (
      <div className="results">
        {meta}
        <article className="clear-card">
          <span className="clear-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="9" />
              <path d="m8.5 12.5 2.5 2.5 4.5-5" />
            </svg>
          </span>
          <h2>No significant findings</h2>
          <p className="clear-message">
            {result.message ||
              `No findings reached the ${thresholdPct}% confidence threshold for this radiograph.`}
          </p>
          <p className="clear-note">
            Only findings at or above {thresholdPct}% confidence are reported. This is not a
            diagnosis — please consult a qualified clinician for interpretation.
          </p>
        </article>
      </div>
    )
  }

  // ---- One or more significant findings ----
  const primary = findings[0]

  return (
    <div className="results">
      {meta}

      <article className={`primary sev-${primary.sev}`}>
        <div className="primary-head">
          <span className="primary-eyebrow">Most likely finding</span>
          <span className={`sev-pill sev-${primary.sev}`}>
            {SEVERITY_LABEL[primary.sev]} confidence
          </span>
        </div>
        <h2 className="primary-name">{primary.name}</h2>
        {primary.desc && <p className="primary-desc">{primary.desc}</p>}
        <div className="primary-score">
          <span className="primary-big">{formatPercent(primary.score)}</span>
          <span className="primary-big-label">model confidence</span>
        </div>
        <div className="primary-track">
          <div
            className="primary-fill"
            style={{ width: animate ? `${barWidth(primary.score)}%` : 0 }}
          />
        </div>
      </article>

      <section className="panel findings">
        <div className="panel-head">
          <h3>Findings above {thresholdPct}%</h3>
          <span className="muted">
            {findings.length} detected
          </span>
        </div>

        <ul className="bars">
          {findings.map((finding, index) => (
            <li className="bar-row" key={finding.key} title={finding.desc}>
              <div className="bar-label">
                <span className={`dot sev-${finding.sev}`} aria-hidden="true" />
                <span className="bar-name">{finding.name}</span>
              </div>
              <div className="bar-track">
                <div
                  className={`bar-fill sev-${finding.sev}`}
                  style={{
                    width: animate ? `${barWidth(finding.score)}%` : 0,
                    transitionDelay: `${index * 35}ms`,
                  }}
                />
              </div>
              <span className="bar-pct">{formatPercent(finding.score)}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
