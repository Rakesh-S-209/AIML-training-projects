import PredictionForm from './components/PredictionForm'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🏠 Housing Price Predictor</h1>
        <p>Enter property details to predict the price category (Low / Medium / High)</p>
      </header>

      <main className="card">
        <PredictionForm />
      </main>
    </div>
  )
}
