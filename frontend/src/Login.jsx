
import './Login.css'

function Login({ onBack, onLogin }) {
  return (
    <div className="login-page">
      <div className="login-card">

        <div className="login-logo">
          <span className="login-crystal">◇</span>
          <span>CRYSTAL</span>
        </div>

        <h1>Welcome Back</h1>

        <p className="login-subtitle">
          Sign in to continue to Voltage Prediction
        </p>

        <form className="login-form"
        onSubmit={(event) => {
    event.preventDefault()
    onLogin()
    }}
     >
          <div className="login-field">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Enter your password"
              required
            />
          </div>

          <button type="submit" className="login-submit">
            Login
          </button>
        </form>

        <p className="signup-text">
          Don't have an account? <button type="button">Sign Up</button>
        </p>

        <button
          type="button"
          className="back-home"
          onClick={onBack}
        >
          ← Back to Home
        </button>

      </div>
    </div>
  )
}

export default Login