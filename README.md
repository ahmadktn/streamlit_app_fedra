# FEDRA 3.0: Nigerian Student Dropout & Fairness Early Warning Dashboard

A multi-page Streamlit application for institutional learning analytics, student failure risk early warning, empirical pattern exploration, and algorithmic fairness auditing on the 50,150 Nigerian student cohort.

---

## Features & Visualizations Suite

1. **Cohort Early Warning Monitor:**
   - Real-time triage queue ranking students by failure/dropout probability.
   - Interactive **Cohort Risk Probability Distribution** histogram with color-coded risk zones and calibrated decision threshold.
   - **Faculty Attrition Risk Breakdown** ranking attrition across all university faculties.
   - Filterable student queue with progress risk indicators and intervention recommendations.

2. **EDA & Empirical Risk Patterns (from Jupyter Notebook):**
   - **Student Academic Outcome Distribution**: Interactive donut breakdown across outcomes (Graduated, Enrolled, Dropout, Suspended, Expelled).
   - **Academic Momentum Box Plots**: GPA velocity ($\Delta = \text{GPA} - \text{Prev GPA}$) distribution across outcomes with zero-momentum reference line.
   - **Attendance vs. GPA Density Scatter**: High-density clustering visualization highlighting the Critical Danger Zone (Attendance < 60% & GPA < 2.0).
   - **Funding Source & Tuition Vulnerability**: Attrition rates by funding mechanism (Self-Funded, Loans, Scholarships) and tuition arrears breakdown.
   - **Faculty & Geographic Disparities**: Regional state-level attrition rates across 20 Nigerian states.

3. **Student Risk Simulator & Explainability Lab:**
   - Interactive **FEDRA 3.0 Risk Gauge Dial** with dynamic threshold needle and color-coded danger zones.
   - **Local TreeSHAP Attribution Waterfall**: Feature-level contribution chart deconstructing positive (risk-increasing) and negative (protective) drivers for the individual student.
   - **What-If Scenario Counterfactuals**: Side-by-side comparative bar chart evaluating the impact of bursaries, attendance boosts, and mentoring across ensemble branches.

4. **Model Benchmarks & Threshold Lab:**
   - Multi-model performance summary across Logistic Regression, Random Forest, XGBoost, CatBoost, and FEDRA 3.0.
   - **Model Comparison Bar Chart**: Side-by-side metric comparison (Accuracy, Precision, Recall, F1, ROC-AUC).
   - **Receiver Operating Characteristic (ROC) Curves**: Full-cohort interactive curves with AUC in legend and hover diagnostics.
   - **Precision-Recall Curves**: Early-warning curves comparing recall against positive predictive value.
   - **Probability Reliability / Calibration Curves**: 10-bin calibration curves comparing predicted probabilities to empirical rates.
   - **Interactive Confusion Matrix Heatmap**: Live-updating color heatmap responding to dynamic threshold slider adjustments.

5. **Fairness Audit & Mitigation Studio:**
   - Multi-attribute audit across **Gender** (Male vs. Female), **Funding Source** (Self-Funded vs. Sponsored), and **Parental Education** (First-Gen vs. Non-First-Gen).
   - **Demographic Disparity Group Comparison Charts**: Visualizing Positive Selection Rate and True Positive Rate (TPR / Sensitivity) disparities.
   - **Pre vs. Post Mitigation Impact Chart**: Comparing DPD, EOD, and model F1 before and after Pareto threshold calibration.

---

## Setup & Running with `uv`

### 1. Sync Dependencies
From the `apps/streamlit_app` directory:
```bash
uv sync
```
Or to install/add the required packages explicitly:
```bash
uv add streamlit pandas numpy scikit-learn catboost xgboost altair plotly
```

### 2. Launch the Streamlit App
```bash
uv run streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.
