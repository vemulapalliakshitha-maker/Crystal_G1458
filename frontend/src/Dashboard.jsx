
import './Dashboard.css'
import "./VoltagePrediction.css";

function Dashboard({ onLogout, onExplore }) {
  return (
    <div className="dashboard-page">
      <header className="dashboard-navbar">
        <div className="dashboard-logo">
          <span className="dashboard-crystal">◇</span>
          <span>CRYSTAL</span>
        </div>

        <button
          type="button"
          className="dashboard-logout"
          onClick={onLogout}
        >
          Logout
        </button>
      </header>

      <main className="dashboard-main">
        <section className="dashboard-welcome">
          <p className="dashboard-label">
            MATERIAL INTELLIGENCE PLATFORM
          </p>

          <h1>
            Welcome to <span>CRYSTAL</span>
          </h1>

          <p className="dashboard-description">
            Machine Learning-Based Voltage Prediction for
            Metal-Ion Battery Electrode Materials.
          </p>
        </section>

        <section className="dashboard-project">
          <div className="dashboard-card-icon">◇</div>

          <p className="dashboard-card-label">
            YOUR PROJECT
          </p>

          <h2>Voltage Prediction</h2>

          <p className="dashboard-card-description">
            Predict the voltage of electrode materials using
            machine learning. Explore material properties and
            identify promising candidates for further research.
          </p>

          <div className="dashboard-models">
            <span>DNN</span>
            <span>SVR</span>
            <span>KRR</span>
          </div>

          <button
            type="button"
            className="dashboard-explore"
            onClick={onExplore}
          >
            Explore Voltage Prediction <span>→</span>
          </button>
        </section>

        <p className="dashboard-footer">
          CRYSTAL · MATERIAL INTELLIGENCE
        </p>
      </main>
    </div>
  )
}

export default Dashboard
