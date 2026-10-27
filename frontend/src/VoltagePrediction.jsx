import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./VoltagePrediction.css";

function VoltagePrediction({ onBack }) {
  const initialForm = {
    metalIon: "Li",
    highFormula: "LiFePO4",
    lowFormula: "FePO4",
    concentration: "1.0",
    lattice: "",
    spaceGroup: "",
  };

  const [form, setForm] = useState(initialForm);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setError("");
  };

  const handleClear = () => {
    setForm({
      metalIon: "Li",
      highFormula: "",
      lowFormula: "",
      concentration: "",
      lattice: "",
      spaceGroup: "",
    });

    setResults(null);
    setError("");
  };

  const handlePredict = async (event) => {
    event.preventDefault();
    setError("");
    setResults(null);

    if (
      !form.highFormula.trim() ||
      !form.lowFormula.trim() ||
      form.concentration === "" ||
      !form.lattice ||
      form.spaceGroup === ""
    ) {
      setError("Please fill in all the fields before predicting.");
      return;
    }

    const concentration = Number(form.concentration);
    const spaceGroup = Number(form.spaceGroup);

    if (!Number.isFinite(concentration) || concentration < 0 || concentration > 1) {
      setError("Metal-ion concentration must be between 0 and 1.");
      return;
    }

    if (!Number.isInteger(spaceGroup) || spaceGroup < 1 || spaceGroup > 230) {
      setError("Space group must be an integer between 1 and 230.");
      return;
    }

    /*
      DEMO MODE:
      These values are illustrative and are NOT predictions from a trained model.

      Later, replace this section with a fetch() request to your FastAPI backend.
      The backend must generate valid composition features, apply the fitted
      preprocessing/PCA pipeline, and run the trained model.
    */

    const demoData = [
      { concentration: 0.2, voltage: 3.18 },
      { concentration: 0.3, voltage: 3.24 },
      { concentration: 0.4, voltage: 3.31 },
      { concentration: 0.5, voltage: 3.38 },
      { concentration: 0.6, voltage: 3.35 },
      { concentration: 0.7, voltage: 3.43 },
      { concentration: 0.8, voltage: 3.51 },
      { concentration: 0.9, voltage: 3.47 },
      { concentration: 1.0, voltage: 3.56 },
    ];

    setResults({
      data: demoData,
      averageVoltage: (
        demoData.reduce((total, point) => total + point.voltage, 0) /
        demoData.length
      ).toFixed(3),
    });
  };

  return (
    <main className="voltage-page">
      <header className="voltage-header">
        <button
          type="button"
          className="voltage-back-button"
          onClick={onBack}
        >
          ← Back to Dashboard
        </button>

        <div className="voltage-brand">
          <span className="voltage-brand-icon">✦</span>
          CRYSTAL
        </div>

        <div className="voltage-header-label">
          MATERIALS INTELLIGENCE
        </div>
      </header>

      <section className="voltage-hero">
        <p className="voltage-eyebrow">AI-POWERED MATERIALS ANALYSIS</p>

        <h1>
          Voltage <span>Prediction</span>
        </h1>

        <p>
          Explore the voltage profile of metal-ion battery electrode materials
          using machine learning.
        </p>
      </section>

      <section className="voltage-input-card">
        <div className="voltage-section-heading">
          <div>
            <span className="voltage-step">01 / INPUT</span>
            <h2>Material Information</h2>
            <p>Enter the properties of your electrode material.</p>
          </div>

          <span className="voltage-card-symbol">◇</span>
        </div>

        <form onSubmit={handlePredict}>
          <div className="voltage-form-grid">
            <div className="voltage-field">
              <label htmlFor="metalIon">Metal-ion type</label>
              <select
                id="metalIon"
                name="metalIon"
                value={form.metalIon}
                onChange={handleChange}
              >
                <option value="Li">Lithium (Li)</option>
                <option value="Mg">Magnesium (Mg)</option>
                <option value="Ca">Calcium (Ca)</option>
                <option value="Al">Aluminum (Al)</option>
                <option value="Zn">Zinc (Zn)</option>
                <option value="Y">Yttrium (Y)</option>
              </select>
              <small>Select the metal ion being inserted or removed.</small>
            </div>

            <div className="voltage-field">
              <label htmlFor="highFormula">
                High-concentration stoichiometry
              </label>
              <input
                id="highFormula"
                name="highFormula"
                value={form.highFormula}
                onChange={handleChange}
                placeholder="e.g. LiFePO4"
              />
              <small>Formula representing the higher-ion-content material.</small>
            </div>

            <div className="voltage-field">
              <label htmlFor="lowFormula">
                Low-concentration stoichiometry
              </label>
              <input
                id="lowFormula"
                name="lowFormula"
                value={form.lowFormula}
                onChange={handleChange}
                placeholder="e.g. FePO4"
              />
              <small>Formula representing the lower-ion-content material.</small>
            </div>

            <div className="voltage-field">
              <label htmlFor="concentration">
                High-concentration ion fraction
              </label>
              <input
                id="concentration"
                name="concentration"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={form.concentration}
                onChange={handleChange}
                placeholder="e.g. 1.0"
              />
              <small>Enter a fraction from 0 to 1.</small>
            </div>

            <div className="voltage-field">
              <label htmlFor="lattice">Crystal lattice type</label>
              <select
                id="lattice"
                name="lattice"
                value={form.lattice}
                onChange={handleChange}
              >
                <option value="">Select lattice type</option>
                <option value="Cubic">Cubic</option>
                <option value="Tetragonal">Tetragonal</option>
                <option value="Orthorhombic">Orthorhombic</option>
                <option value="Hexagonal">Hexagonal</option>
                <option value="Monoclinic">Monoclinic</option>
                <option value="Triclinic">Triclinic</option>
                <option value="Trigonal">Trigonal</option>
              </select>
              <small>
                Choose the material's crystal system. Match the backend encoding.
              </small>
            </div>

            <div className="voltage-field">
              <label htmlFor="spaceGroup">Space group number</label>
              <input
                id="spaceGroup"
                name="spaceGroup"
                type="number"
                min="1"
                max="230"
                step="1"
                value={form.spaceGroup}
                onChange={handleChange}
                placeholder="e.g. 221"
              />
              <small>Use the material's actual space group (1–230).</small>
            </div>
          </div>

          {error && (
            <div className="voltage-error" role="alert">
              {error}
            </div>
          )}

          <div className="voltage-form-actions">
            <button type="submit" className="voltage-predict-button">
              Predict Voltage <span>→</span>
            </button>

            <button
              type="button"
              className="voltage-clear-button"
              onClick={handleClear}
            >
              Clear Inputs
            </button>
          </div>
        </form>
      </section>

      {results && (
        <section className="voltage-results">
          <div className="voltage-section-heading">
            <div>
              <span className="voltage-step">02 / RESULTS</span>
              <h2>Predicted Voltage Profile</h2>
              <p>
                Voltage profile visualization for the submitted material.
              </p>
            </div>

            <span className="voltage-result-badge">DEMO MODE</span>
          </div>

          <div className="voltage-demo-notice">
            <strong>Demonstration only:</strong> The graph below uses
            illustrative values. It is not a real ML prediction. Connect the
            FastAPI backend and trained model before using results scientifically.
          </div>

          <div className="voltage-material-summary">
            <div>
              <span>METAL ION</span>
              <strong>{form.metalIon}</strong>
            </div>

            <div>
              <span>HIGH-CONCENTRATION FORMULA</span>
              <strong>{form.highFormula}</strong>
            </div>

            <div>
              <span>LOW-CONCENTRATION FORMULA</span>
              <strong>{form.lowFormula}</strong>
            </div>

            <div>
              <span>AVERAGE DEMO VOLTAGE</span>
              <strong>{results.averageVoltage} V</strong>
            </div>
          </div>

          <div className="voltage-chart-card">
            <div className="voltage-chart-heading">
              <div>
                <h3>Voltage vs. Metal-Ion Concentration</h3>
                <p>Hover over a point to view its illustrative value.</p>
              </div>

              <span className="voltage-chart-unit">VOLTAGE / V</span>
            </div>

            <div className="voltage-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={results.data}
                  margin={{ top: 20, right: 20, left: 5, bottom: 15 }}
                >
                  <CartesianGrid
                    stroke="#263b40"
                    strokeDasharray="4 5"
                  />

                  <XAxis
                    dataKey="concentration"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: "#a2b8ba", fontSize: 12 }}
                    tickLine={false}
                    axisLine={{ stroke: "#3a5155" }}
                    label={{
                      value: "Metal-ion concentration",
                      position: "insideBottom",
                      offset: -5,
                      fill: "#a2b8ba",
                    }}
                  />

                  <YAxis
                    domain={["auto", "auto"]}
                    tick={{ fill: "#a2b8ba", fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                    label={{
                      value: "Voltage (V)",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#a2b8ba",
                    }}
                  />

                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#101d21",
                      border: "1px solid #2c6664",
                      borderRadius: "10px",
                      color: "#f0f7f5",
                    }}
                    labelFormatter={(value) =>
                      `Concentration: ${Number(value).toFixed(2)}`
                    }
                    formatter={(value) => [
                      `${Number(value).toFixed(3)} V`,
                      "Demo voltage",
                    ]}
                  />

                  <Line
                    type="monotone"
                    dataKey="voltage"
                    stroke="#43e0bc"
                    strokeWidth={3}
                    dot={{
                      r: 4,
                      fill: "#43e0bc",
                      stroke: "#101d21",
                      strokeWidth: 2,
                    }}
                    activeDot={{
                      r: 7,
                      fill: "#ffbf69",
                      stroke: "#ffffff",
                      strokeWidth: 2,
                    }}
                    isAnimationActive
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="voltage-result-note">
            <span>ⓘ</span>
            <p>
              The final scientific results must come from the trained model
              using the same feature generation, normalization, and PCA
              transformations used during training.
            </p>
          </div>
        </section>
      )}

      <footer className="voltage-footer">
        CRYSTAL · MACHINE-LEARNED MATERIALS ANALYSIS
      </footer>
    </main>
  );
}

export default VoltagePrediction;