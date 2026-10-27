
import './App.css'
import { useState } from 'react'
import Login from './Login'
import Dashboard from './Dashboard'
import VoltagePrediction from './VoltagePrediction'

// Small gem-shaped logo, drawn as inline SVG
function CrystalIcon() {
  return (
    <svg className="logo-icon" viewBox="0 0 32 32" aria-hidden="true">
      <polygon points="16,2 28,10 28,22 16,30 4,22 4,10" />
      <polyline points="4,10 16,16 28,10" />
      <line x1="16" y1="16" x2="16" y2="30" />
    </svg>
  )
}

// Voltage readings shown beside the battery
const voltageTicks = ['4.0 V', '3.0 V', '2.0 V', '1.0 V']

// Project highlights
const highlights = [
  { title: 'AI / ML', text: 'Voltage Prediction' },
  { title: 'DNN · SVR · KRR', text: 'Machine Learning Models' },
  { title: 'Metal-Ion Batteries', text: 'Material Research' },
]

function App() {
  const [showLogin, setShowLogin] = useState(false)
  const [showDashboard, setShowDashboard] = useState(false)
  const [showVoltagePrediction, setShowVoltagePrediction] = useState(false)

  // Page 4: Voltage Prediction
  if (showVoltagePrediction) {
    return (
      <VoltagePrediction
        onBack={() => setShowVoltagePrediction(false)}
      />
    )
  }

  // Page 3: Dashboard
  if (showDashboard) {
    return (
      <Dashboard
        onLogout={() => {
          setShowDashboard(false)
          setShowLogin(false)
          setShowVoltagePrediction(false)
        }}
        onExplore={() => {
          setShowVoltagePrediction(true)
        }}
      />
    )
  }

  // Page 2: Login
  if (showLogin) {
    return (
      <Login
        onBack={() => setShowLogin(false)}
        onLogin={() => {
          setShowLogin(false)
          setShowDashboard(true)
        }}
      />
    )
  }

  // Page 1: Landing page
  return (
    <div className="landing-page">
      {/* Navigation */}
      <header className="navbar">
        <a className="logo" href="#top" aria-label="CRYSTAL home">
          <CrystalIcon />
          <span className="logo-text">CRYSTAL</span>
        </a>

        <nav className="nav-buttons" aria-label="Account">
          <button
            type="button"
            className="btn btn-primary btn-small"
            onClick={() => setShowLogin(true)}
          >
            Login
          </button>
        </nav>
      </header>

      <main id="top" className="main">
        {/* Hero section */}
        <section className="hero">
          <div className="hero-content">
            <p className="status-badge">
              <span className="status-dot" aria-hidden="true" />
              MACHINE LEARNING · MATERIAL SCIENCE
            </p>

            <h1 className="hero-title">
              Machine Learning the Voltage of{' '}
              <span className="accent">Electrode Materials</span> in Metal-ion
              Batteries.
            </h1>

            <p className="hero-text">
              Predict voltage profiles of metal-ion battery electrode materials
              using machine learning. Explore materials faster and identify
              promising candidates for further investigation.
            </p>

            <div className="hero-buttons"></div>
          </div>

          {/* Battery illustration */}
          <div className="battery-container" aria-hidden="true">
            <div className="battery-glow" />
            <div className="orbit orbit-1" />
            <div className="orbit orbit-2" />
            <div className="orbit orbit-3" />

            <div className="battery">
              <div className="voltage-scale">
                {voltageTicks.map((tick) => (
                  <span className="voltage-tick" key={tick}>
                    {tick}
                  </span>
                ))}
              </div>

              <div className="battery-cap" />

              <div className="battery-body">
                <div className="battery-level" />

                <svg className="battery-bolt" viewBox="0 0 24 24">
                  <path d="M13.5 2 4.5 13.5h6L9.5 22l9-11.5h-6z" />
                </svg>
              </div>
            </div>

            <span className="floating-label label-top">
              Voltage Prediction
            </span>

            <span className="floating-label label-bottom">
              Machine Learning
            </span>
          </div>
        </section>

        {/* Project highlights */}
        <section className="project-info" aria-label="Project highlights">
          {highlights.map((item) => (
            <div className="info-item" key={item.title}>
              <strong className="info-title">{item.title}</strong>
              <span className="info-text">{item.text}</span>
            </div>
          ))}
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>CRYSTAL · MATERIAL INTELLIGENCE</span>
        <span>Machine-Learned Voltage Prediction</span>
      </footer>
    </div>
  )
}

export default App
