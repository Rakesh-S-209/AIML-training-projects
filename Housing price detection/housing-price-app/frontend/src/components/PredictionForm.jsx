import { useState } from 'react'

// ── Default form state ────────────────────────────────────────────────────
const INITIAL_FORM = {
  area: '',
  bedrooms: '',
  bathrooms: '',
  stories: '',
  parking: '',
  mainroad: 'yes',
  guestroom: 'no',
  basement: 'no',
  hotwaterheating: 'no',
  airconditioning: 'no',
  prefarea: 'no',
  furnishingstatus: 'furnished',
}

// ── Label helpers ─────────────────────────────────────────────────────────
const CATEGORY_CLASS = { Low: 'low', Medium: 'medium', High: 'high' }

const PROB_ORDER = ['Low', 'Medium', 'High']

// ── Component ─────────────────────────────────────────────────────────────
export default function PredictionForm() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [result, setResult] = useState(null)   // { label, probabilities }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // ── Handlers ─────────────────────────────────────────────────────────────
  function handleChange(e) {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
    // Clear previous result when user edits inputs
    setResult(null)
    setError(null)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          // Numeric fields must be numbers, not strings
          area: Number(form.area),
          bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms),
          stories: Number(form.stories),
          parking: Number(form.parking),
        }),
      })

      const json = await res.json()

      if (!res.ok) {
        setError(json.error || 'Prediction failed. Please try again.')
      } else {
        setResult(json)
      }
    } catch (err) {
      setError('Could not reach the server. Is the Flask backend running on port 5000?')
    } finally {
      setLoading(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <form onSubmit={handleSubmit} noValidate>

      {/* ── Numeric features ─────────────────────────────────────────── */}
      <p className="section-title">Property Details</p>
      <div className="form-grid">
        <NumericField
          label="Area (sq ft)"
          name="area"
          value={form.area}
          onChange={handleChange}
          min={500}
          placeholder="e.g. 7420"
        />
        <NumericField
          label="Bedrooms"
          name="bedrooms"
          value={form.bedrooms}
          onChange={handleChange}
          min={1}
          max={10}
          placeholder="e.g. 3"
        />
        <NumericField
          label="Bathrooms"
          name="bathrooms"
          value={form.bathrooms}
          onChange={handleChange}
          min={1}
          max={10}
          placeholder="e.g. 2"
        />
        <NumericField
          label="Stories"
          name="stories"
          value={form.stories}
          onChange={handleChange}
          min={1}
          max={10}
          placeholder="e.g. 2"
        />
        <NumericField
          label="Parking Spaces"
          name="parking"
          value={form.parking}
          onChange={handleChange}
          min={0}
          max={10}
          placeholder="e.g. 1"
        />
      </div>

      {/* ── Yes / No features ────────────────────────────────────────── */}
      <p className="section-title">Amenities</p>
      <div className="form-grid">
        <YesNoField label="Main Road Access"    name="mainroad"        value={form.mainroad}        onChange={handleChange} />
        <YesNoField label="Guest Room"          name="guestroom"       value={form.guestroom}       onChange={handleChange} />
        <YesNoField label="Basement"            name="basement"        value={form.basement}        onChange={handleChange} />
        <YesNoField label="Hot Water Heating"   name="hotwaterheating" value={form.hotwaterheating} onChange={handleChange} />
        <YesNoField label="Air Conditioning"    name="airconditioning" value={form.airconditioning} onChange={handleChange} />
        <YesNoField label="Preferred Area"      name="prefarea"        value={form.prefarea}        onChange={handleChange} />
      </div>

      {/* ── Furnishing status ────────────────────────────────────────── */}
      <p className="section-title">Furnishing</p>
      <div className="form-grid">
        <div className="field">
          <label htmlFor="furnishingstatus">Furnishing Status</label>
          <select
            id="furnishingstatus"
            name="furnishingstatus"
            value={form.furnishingstatus}
            onChange={handleChange}
          >
            <option value="furnished">Furnished</option>
            <option value="semi-furnished">Semi-furnished</option>
            <option value="unfurnished">Unfurnished</option>
          </select>
        </div>
      </div>

      {/* ── Submit ───────────────────────────────────────────────────── */}
      <button type="submit" className="btn-predict" disabled={loading}>
        {loading ? 'Predicting…' : 'Predict Price Category'}
      </button>

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && <p className="error-msg">⚠ {error}</p>}

      {/* ── Result ───────────────────────────────────────────────────── */}
      {result && <ResultPanel result={result} />}
    </form>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────
function NumericField({ label, name, value, onChange, min, max, placeholder }) {
  return (
    <div className="field">
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        placeholder={placeholder}
        required
      />
    </div>
  )
}

function YesNoField({ label, name, value, onChange }) {
  return (
    <div className="field">
      <label htmlFor={name}>{label}</label>
      <select id={name} name={name} value={value} onChange={onChange}>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </div>
  )
}

function ResultPanel({ result }) {
  const cls = CATEGORY_CLASS[result.label] ?? 'low'

  return (
    <div className={`result-panel ${cls}`}>
      <p className="result-label">Predicted Price Category</p>
      <p className="result-category">{result.label}</p>

      {result.probabilities && (
        <div className="probabilities">
          {PROB_ORDER.map(cat => (
            <div className="prob-row" key={cat}>
              <span className="prob-row-label">{cat}</span>
              <div className="prob-bar-track">
                <div
                  className={`prob-bar-fill ${cat.toLowerCase()}`}
                  style={{ width: `${result.probabilities[cat]}%` }}
                />
              </div>
              <span className="prob-pct">{result.probabilities[cat]}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
