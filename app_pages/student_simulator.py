import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fedra_app.pipeline import train_suite_models, predict_single_student, compute_student_shap, FACULTIES, STATES
from fedra_app.visualizations import plot_student_risk_gauge, plot_student_shap_waterfall, plot_what_if_comparison

st.header("Single-student risk simulator & what-if lab", anchor=False)
st.caption("Diagnose individual risk trajectories, simulate scenario interventions, and view real-time risk adjustments for Nigerian students.")

artifacts = train_suite_models()
meta_weights = artifacts['meta_weights']
th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']

# Presets for quick evaluation
preset = st.segmented_control(
    "Load student behavioral archetype",
    [
        "Custom student",
        "The academic crisis student",
        "The resilient high-performer",
        "The financially strained commuter",
        "The mentored turnaround"
    ],
    default="The academic crisis student"
)

# Preset configs
if preset == "The academic crisis student":
    def_gpa, def_prev_gpa, def_att, def_study, def_failed = 1.8, 3.5, 45.0, 6.0, 4.0
    def_tuition, def_funding, def_allowance, def_dist, def_mentor = "Unpaid", "Self Funded", 15000.0, 180.0, 0
elif preset == "The resilient high-performer":
    def_gpa, def_prev_gpa, def_att, def_study, def_failed = 4.2, 4.1, 88.0, 28.0, 0.0
    def_tuition, def_funding, def_allowance, def_dist, def_mentor = "Paid", "Full Scholarship", 55000.0, 25.0, 4
elif preset == "The financially strained commuter":
    def_gpa, def_prev_gpa, def_att, def_study, def_failed = 2.4, 2.9, 58.0, 12.0, 2.0
    def_tuition, def_funding, def_allowance, def_dist, def_mentor = "Unpaid", "Self Funded", 18000.0, 360.0, 1
elif preset == "The mentored turnaround":
    def_gpa, def_prev_gpa, def_att, def_study, def_failed = 3.3, 2.1, 78.0, 22.0, 1.0
    def_tuition, def_funding, def_allowance, def_dist, def_mentor = "Paid", "Partial Scholarship", 42000.0, 60.0, 8
else:
    def_gpa, def_prev_gpa, def_att, def_study, def_failed = 3.0, 3.0, 70.0, 18.0, 1.0
    def_tuition, def_funding, def_allowance, def_dist, def_mentor = "Paid", "Self Funded", 35000.0, 100.0, 2

col_input, col_output = st.columns([1.1, 0.9])

with col_input:
    with st.container(border=True):
        st.subheader("Academic trajectory & institutional background", anchor=False)
        
        c1, c2 = st.columns(2)
        with c1:
            sel_faculty = st.selectbox("Faculty", FACULTIES, index=0)
            sel_gender = st.selectbox("Gender", ["Male", "Female"])
            sel_year = st.selectbox("Year of study", [1, 2, 3, 4, 5], index=1)
            sel_tuition = st.selectbox("Tuition fee status", ["Paid", "Unpaid"], index=0 if def_tuition=="Paid" else 1)
        with c2:
            sel_state = st.selectbox("State of origin", STATES, index=STATES.index("Lagos") if "Lagos" in STATES else 0)
            sel_funding = st.selectbox("Funding source", ["Self Funded", "Government Loan", "Partial Scholarship", "Full Scholarship", "Family Support"], index=0 if def_funding=="Self Funded" else 2)
            sel_housing = st.selectbox("Housing accommodation", ["Off Campus", "On Campus", "Family Home"], index=0)
            sel_allowance = st.number_input("Monthly allowance (₦ Naira)", 5000.0, 150000.0, float(def_allowance), step=5000.0)

        st.caption("Academic milestones & behavioral indicators")
        c3, c4 = st.columns(2)
        with c3:
            inp_gpa = st.slider("Current estimated GPA (out of 5.0)", 0.0, 5.0, float(def_gpa), step=0.05)
            inp_prev_gpa = st.slider("Previous semester GPA", 0.0, 5.0, float(def_prev_gpa), step=0.05)
            inp_att = st.slider("Lecture attendance rate (%)", 10.0, 100.0, float(def_att), step=1.0)
        with c4:
            inp_failed = st.number_input("Failed courses registered", 0, 8, int(def_failed))
            inp_study = st.slider("Weekly independent study hours", 0, 50, int(def_study))
            inp_dist = st.number_input("Commute distance from home (km)", 0, 600, int(def_dist), step=20)
            inp_mentor = st.slider("Mentoring sessions attended", 0, 12, int(def_mentor))

# Student dictionary
sim_student = {
    'Faculty': sel_faculty,
    'State': sel_state,
    'Gender': sel_gender,
    'YearOfStudy': sel_year,
    'TuitionPaid': sel_tuition,
    'FundingSource': sel_funding,
    'Housing': sel_housing,
    'MonthlyAllowance': sel_allowance,
    'GPA': inp_gpa,
    'PreviousSemesterGPA': inp_prev_gpa,
    'AttendanceRate': inp_att,
    'FailedCourses': inp_failed,
    'StudyHoursPerWeek': inp_study,
    'DistanceFromHome': inp_dist,
    'MentoringSessionsAttended': inp_mentor,
    'CoursesCreditLoad': 18.0,
    'Dependants': 1,
    'CounsellingVisits': 0
}

res = predict_single_student(sim_student, artifacts=artifacts)
risk_score = res['risk_score']
risk_pct = res['risk_score_pct']
is_at_risk = res['is_at_risk']
priority = res['priority']

with col_output:
    with st.container(border=True):
        st.subheader("FEDRA 3.0 Early-Warning Assessment", anchor=False)
        
        # Priority Alert Banner
        if priority == "Critical Priority":
            st.error(f"🚨 **CRITICAL ATTRITION RISK** | Estimated Dropout Probability: **{risk_pct:.1f}%**")
        elif priority == "Moderate Risk":
            st.warning(f"⚠️ **MODERATE RISK DETECTED** | Estimated Dropout Probability: **{risk_pct:.1f}%**")
        else:
            st.success(f"✅ **ON TRACK / SATISFACTORY** | Estimated Dropout Probability: **{risk_pct:.1f}%**")

        # 1. Plotly Risk Gauge Meter
        fig_gauge = plot_student_risk_gauge(risk_pct, threshold_pct=th_fedra * 100)
        st.plotly_chart(fig_gauge, width='stretch')
        
        # Branch Breakdown Metrics
        with st.container(horizontal=True):
            st.metric("CatBoost Branch", f"{res['breakdown']['CatBoost']:.1f}%")
            st.metric("XGBoost Branch", f"{res['breakdown']['XGBoost']:.1f}%")
            st.metric("Logistic Anchor", f"{res['breakdown']['Logistic Regression']:.1f}%")

st.divider()

# Diagnostic tabs: SHAP Local Explainability & What-If Lab
tab_shap, tab_whatif = st.tabs([
    "Local risk factor attributions (TreeSHAP)",
    "What-if scenario & intervention simulation"
])

with tab_shap:
    st.markdown("##### Individual Student Risk Factor Contributions")
    st.caption("Deconstructs the exact positive (risk-increasing) and negative (protective) feature attributions driving this student's profile.")
    
    with st.spinner("Computing local SHAP attributions..."):
        df_shap, base_val, _ = compute_student_shap(sim_student, artifacts=artifacts, max_features=12)
        
        col_sh_c, col_sh_t = st.columns([1.3, 1])
        with col_sh_c:
            fig_shap = plot_student_shap_waterfall(df_shap, base_val=base_val)
            st.plotly_chart(fig_shap, use_container_width=True)
            
        with col_sh_t:
            st.markdown("**SHAP Feature Impact Breakdown Table**")
            tbl_shap = df_shap[['label', 'shap_value', 'direction', 'raw_value']].copy()
            tbl_shap['raw_value'] = tbl_shap['raw_value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
            st.dataframe(
                tbl_shap,
                column_config={
                    "label": "Risk Factor",
                    "raw_value": "Student Value",
                    "shap_value": st.column_config.NumberColumn("Impact (Log-Odds)", format="%+.3f"),
                    "direction": "Risk Effect"
                },
                hide_index=True,
                use_container_width=True
            )
            
    st.caption("Red bars indicate factors exacerbating student attrition risk. Green bars represent protective academic or demographic factors.")

with tab_whatif:
    st.markdown("##### Proactive Intervention Simulation")
    st.caption("Simulate real-time student trajectory changes under targeted institutional support actions.")
    
    col_scen_ctrl, col_scen_chart = st.columns([1, 1.3])
    
    with col_scen_ctrl:
        with st.container(border=True):
            st.markdown("**Intervention Levers**")
            interv_tuition = st.checkbox("Award emergency student bursary (Tuition -> Paid)", value=False)
            interv_att = st.slider("Proactive academic coaching & check-ins (+% Attendance)", 0, 40, 15, step=5)
            interv_mentor = st.slider("Assign peer mentor (+ Mentoring sessions)", 0, 8, 2)
            
            # Calculate counterfactual
            cf_student = sim_student.copy()
            if interv_tuition:
                cf_student['TuitionPaid'] = "Paid"
            cf_student['AttendanceRate'] = min(100.0, cf_student['AttendanceRate'] + interv_att)
            cf_student['MentoringSessionsAttended'] += interv_mentor
            
            cf_res = predict_single_student(cf_student, artifacts=artifacts)
            delta_risk = cf_res['risk_score_pct'] - risk_pct
            
            st.divider()
            st.metric(
                "Projected post-intervention risk score",
                f"{cf_res['risk_score_pct']:.1f}%",
                f"{delta_risk:+.1f}% risk reduction" if delta_risk != 0 else "No change",
                delta_color="inverse",
                border=True
            )
            
            if cf_res['priority'] != priority:
                st.info(f"Trajectory shifts from **{priority}** to **{cf_res['priority']}**!")

    with col_scen_chart:
        fig_whatif = plot_what_if_comparison(res, cf_res)
        st.plotly_chart(fig_whatif, use_container_width=True)
        
    st.divider()
    st.markdown("##### Counterfactual Intervention Impact Table")
    
    cf_table = pd.DataFrame([
        {
            'Dimension': 'Tuition Status',
            'Current Baseline': sim_student['TuitionPaid'],
            'Post-Intervention': cf_student['TuitionPaid'],
            'Change': 'Tuition Resolved' if interv_tuition else 'Unchanged'
        },
        {
            'Dimension': 'Attendance Rate',
            'Current Baseline': f"{sim_student['AttendanceRate']:.1f}%",
            'Post-Intervention': f"{cf_student['AttendanceRate']:.1f}%",
            'Change': f"+{interv_att}%" if interv_att > 0 else 'Unchanged'
        },
        {
            'Dimension': 'Mentoring Sessions',
            'Current Baseline': f"{sim_student['MentoringSessionsAttended']} sessions",
            'Post-Intervention': f"{cf_student['MentoringSessionsAttended']} sessions",
            'Change': f"+{interv_mentor} sessions" if interv_mentor > 0 else 'Unchanged'
        },
        {
            'Dimension': 'Blended Risk Probability (FEDRA 3.0)',
            'Current Baseline': f"{risk_pct:.1f}%",
            'Post-Intervention': f"{cf_res['risk_score_pct']:.1f}%",
            'Change': f"{delta_risk:+.1f}%"
        },
        {
            'Dimension': 'Actionable Triage Tier',
            'Current Baseline': priority,
            'Post-Intervention': cf_res['priority'],
            'Change': 'Risk Downgraded' if cf_res['priority'] != priority else 'Same Tier'
        }
    ])
    st.dataframe(cf_table, hide_index=True, use_container_width=True)


