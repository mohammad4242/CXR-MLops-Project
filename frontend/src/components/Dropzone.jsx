import { useRef, useState } from 'react'
import { formatBytes } from '../lib/format'

// Upload card: drag-and-drop or click-to-browse, image preview, and the
// primary "Analyze" action. Keeps no business logic — it reports the chosen
// File up to App and renders whatever state App hands back.
export default function Dropzone({ file, previewUrl, loading, onSelect, onClear, onAnalyze }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const openPicker = () => inputRef.current?.click()

  const handleInputChange = (event) => {
    onSelect(event.target.files?.[0] ?? null)
    // Reset so selecting the same file again still fires onChange.
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragOver(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) onSelect(dropped)
  }

  return (
    <section className="panel dropzone-panel">
      <div className="panel-head">
        <h2>Upload X-ray</h2>
        {file && (
          <button className="link-btn" type="button" onClick={onClear} disabled={loading}>
            Remove
          </button>
        )}
      </div>

      {!previewUrl ? (
        <div
          className={`dropzone ${dragOver ? 'is-drag' : ''}`}
          role="button"
          tabIndex={0}
          onClick={openPicker}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              openPicker()
            }
          }}
          onDragOver={(event) => {
            event.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <span className="dz-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4" />
              <path d="m7 9 5-5 5 5" />
              <path d="M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" />
            </svg>
          </span>
          <p className="dz-title">Drag &amp; drop your image here</p>
          <p className="dz-sub">
            or <span>browse files</span> · PNG or JPEG
          </p>
        </div>
      ) : (
        <figure className="preview">
          <img src={previewUrl} alt="Selected chest X-ray preview" />
          <figcaption className="preview-meta">
            <span className="file-name" title={file?.name}>
              {file?.name}
            </span>
            <span className="file-size">{formatBytes(file?.size)}</span>
          </figcaption>
        </figure>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={handleInputChange}
      />

      <button
        className="cta"
        type="button"
        onClick={onAnalyze}
        disabled={!file || loading}
      >
        {loading ? (
          <>
            <span className="spinner" aria-hidden="true" />
            Analyzing…
          </>
        ) : (
          'Analyze X-ray'
        )}
      </button>
    </section>
  )
}
