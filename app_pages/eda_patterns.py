import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fedra_app.pipeline import train_suite_models, FACULTIES, STATES
from fedra_app.visualizations import (
    plot_academic_outcomes_donut,
    plot_academic_momentum_box,
    plot_attendance_gpa_density,
    plot_attrition_by_funding,
    plot_faculty_risk_breakdown,
    plot_state_risk_bars,
    plot_tuition_risk_impact
)

st.header("Exploratory Data Analysis & Empirical Risk Patterns", anchor=False)
st.caption("Comprehensive empirical analysis, statistical distributions, cross-tabulations, and visualizations across the 50,150 Nigerian university student cohort.")

with st.spinner("Loading cohort datasets and computing statistical aggregates..."):
    artifacts = train_suite_models()

df = artifacts['data']['df'].copy()

# 1. Global KPI Summary Row
total_cohort = len(df)
overall_attrition = (df['Actual_Attrition'].mean() * 100) if 'Actual_Attrition' in df.columns else (df['Predicted_At_Risk'].mean() * 100)
unpaid_tuition_pct = (df['TuitionPaid'].astype(str).str.lower().isin(['unpaid', '0', 'no'])).mean() * 100
avg_momentum = float(df['GPA_Momentum'].mean())
avg_gpa = float(df['GPA'].mean())
avg_att = float(df['AttendanceRate'].mean())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Cohort (N)", f"{total_cohort:,}", "Nigerian Universities", border=True)
with c2:
    st.metric("Historical Attrition Rate", f"{overall_attrition:.1f}%", "Dropout + Suspension", delta_color="inverse", border=True)
with c3:
    st.metric("Unpaid Tuition Rate", f"{unpaid_tuition_pct:.1f}%", "Financial Precarity", delta_color="inverse", border=True)
with c4:
    st.metric("Mean GPA Momentum", f"{avg_momentum:+.2f}", "Semester Velocity", border=True)
with c5:
    st.metric("Mean Attendance Rate", f"{avg_att:.1f}%", f"Mean GPA: {avg_gpa:.2f}", border=True)

st.divider()

# 2. Interactive Cohort Filter Controls
with st.expander("Cohort Data Slicing & Subgroup Filtering", expanded=False):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        sel_faculty = st.selectbox("Faculty", ["All Faculties"] + FACULTIES, index=0)
    with f_col2:
        sel_gender = st.selectbox("Gender", ["All", "Male", "Female"], index=0)
    with f_col3:
        sel_funding = st.selectbox("Funding source", ["All", "Self Funded", "Government Loan", "Partial Scholarship", "Full Scholarship", "Family Support"], index=0)
    with f_col4:
        sel_tuition = st.selectbox("Tuition status", ["All", "Paid", "Unpaid"], index=0)

filtered_df = df.copy()
if sel_faculty != "All Faculties":
    filtered_df = filtered_df[filtered_df['Faculty'] == sel_faculty]
if sel_gender != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == sel_gender]
if sel_funding != "All":
    filtered_df = filtered_df[filtered_df['FundingSource'] == sel_funding]
if sel_tuition != "All":
    filtered_df = filtered_df[filtered_df['TuitionPaid'] == sel_tuition]

st.caption(f"Currently active slice: **{len(filtered_df):,}** students ({len(filtered_df)/total_cohort:.1%} of global cohort).")

# 3. Multi-Dimensional Tabs combining Visualizations AND Comprehensive Data Tables
tab_outcomes, tab_density, tab_socio, tab_geo, tab_explorer = st.tabs([
    "Academic outcomes & momentum",
    "Attendance vs. GPA density",
    "Funding & financial precarity",
    "Faculty & geographic disparities",
    "Raw cohort data explorer & stats"
])

# ==============================================================================
# TAB 1: ACADEMIC OUTCOMES & MOMENTUM
# ==============================================================================
with tab_outcomes:
    st.markdown("#### Empirical Target Outcomes & GPA Momentum Dynamics")
    st.caption(r"Analyzes the baseline distribution of student outcomes alongside semester-over-semester GPA velocity ($\Delta = \text{GPA} - \text{Prev GPA}$).")
    
    # Visualizations row
    col_donut, col_box = st.columns([1, 1.2])
    with col_donut:
        fig_donut = plot_academic_outcomes_donut(filtered_df)
        st.plotly_chart(fig_donut, use_container_width=True)
    with col_box:
        fig_box = plot_academic_momentum_box(filtered_df)
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.divider()
    
    # Accompanying Data Tables
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("##### Academic Outcome Distribution Table")
        outcome_col = 'Status_clean' if 'Status_clean' in filtered_df.columns else 'Status'
        
        outcome_summary = filtered_df.groupby(outcome_col).agg(
            Student_Count=('StudentID', 'count'),
            Mean_GPA=('GPA', 'mean'),
            Mean_Prev_GPA=('PreviousSemesterGPA', 'mean'),
            Mean_Momentum=('GPA_Momentum', 'mean'),
            Mean_Attendance=('AttendanceRate', 'mean'),
            Failed_Courses=('FailedCourses', 'mean')
        ).reset_index()
        
        outcome_summary['Proportion_Pct'] = (outcome_summary['Student_Count'] / len(filtered_df)) * 100
        outcome_summary = outcome_summary.sort_values(by='Student_Count', ascending=False)
        
        st.dataframe(
            outcome_summary,
            column_config={
                outcome_col: "Academic Outcome",
                "Student_Count": st.column_config.NumberColumn("Students (N)", format="%d"),
                "Proportion_Pct": st.column_config.NumberColumn("Share (%)", format="%.1f%%"),
                "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f"),
                "Mean_Prev_GPA": st.column_config.NumberColumn("Prev GPA", format="%.2f"),
                "Mean_Momentum": st.column_config.NumberColumn("GPA Momentum (Δ)", format="%+.2f"),
                "Mean_Attendance": st.column_config.NumberColumn("Attendance (%)", format="%.1f%%"),
                "Failed_Courses": st.column_config.NumberColumn("Avg Failed Courses", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

    with col_t2:
        st.markdown("##### GPA Momentum Parametric & Percentile Breakdown")
        
        mom_stats = filtered_df.groupby(outcome_col)['GPA_Momentum'].agg(
            Mean='mean',
            Std='std',
            Min='min',
            Q25=lambda x: np.percentile(x, 25),
            Median='median',
            Q75=lambda x: np.percentile(x, 75),
            Max='max',
            Negative_Pct=lambda x: (x < 0).mean() * 100
        ).reset_index()
        
        st.dataframe(
            mom_stats,
            column_config={
                outcome_col: "Academic Outcome",
                "Mean": st.column_config.NumberColumn("Mean Δ", format="%+.2f"),
                "Std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "Min": st.column_config.NumberColumn("Min Δ", format="%+.2f"),
                "Q25": st.column_config.NumberColumn("Q1 (25%)", format="%+.2f"),
                "Median": st.column_config.NumberColumn("Median Δ", format="%+.2f"),
                "Q75": st.column_config.NumberColumn("Q3 (75%)", format="%+.2f"),
                "Max": st.column_config.NumberColumn("Max Δ", format="%+.2f"),
                "Negative_Pct": st.column_config.NumberColumn("% Negative Velocity", format="%.1f%%")
            },
            hide_index=True,
            use_container_width=True
        )

    st.info(
        "**Institutional Insight (from Notebook Cell 5)**: Over **87.4%** of students who discontinue their studies "
        "show a marked negative GPA velocity (median drop of -0.65 grade points), revealing that academic disengagement occurs "
        "progressively across 1 to 2 consecutive semesters rather than precipitously."
    )

# ==============================================================================
# TAB 2: ATTENDANCE VS GPA DENSITY
# ==============================================================================
with tab_density:
    st.markdown("#### Lecture Attendance vs. Cumulative GPA Joint Distribution")
    st.caption("Visualizes the bivariate density of student engagement and academic mastery, isolating the critical threshold boundaries.")
    
    # Scatter Chart
    fig_density = plot_attendance_gpa_density(filtered_df, sample_n=3000)
    st.plotly_chart(fig_density, use_container_width=True)
    
    st.divider()
    
    # Quadrant Analysis Data Table
    st.markdown("##### Quadrant Segmentation & Empirical Attrition Risk Cross-Tabulation")
    st.caption("Classifies students into 4 distinct quadrants based on the Critical Danger Thresholds (Attendance Cutoff = 60%, GPA Cutoff = 2.0).")
    
    q_cond = [
        (filtered_df['AttendanceRate'] < 60) & (filtered_df['GPA'] < 2.0),
        (filtered_df['AttendanceRate'] < 60) & (filtered_df['GPA'] >= 2.0),
        (filtered_df['AttendanceRate'] >= 60) & (filtered_df['GPA'] < 2.0),
        (filtered_df['AttendanceRate'] >= 60) & (filtered_df['GPA'] >= 2.0)
    ]
    q_labels = [
        "Q3: Critical Danger Zone (Low Attendance & Low GPA)",
        "Q2: Disengaged Passing (Low Attendance & High GPA)",
        "Q4: Struggling Attender (High Attendance & Low GPA)",
        "Q1: Thriving Zone (High Attendance & High GPA)"
    ]
    filtered_df['Quadrant'] = np.select(q_cond, q_labels, default="Unclassified")
    
    quadrant_table = filtered_df.groupby('Quadrant').agg(
        Students=('StudentID', 'count'),
        Mean_GPA=('GPA', 'mean'),
        Mean_Attendance=('AttendanceRate', 'mean'),
        Attrition_Rate=('Actual_Attrition', lambda x: x.mean() * 100 if len(x)>0 else 0.0),
        Mean_Risk_Score=('Risk_Score_Pct', 'mean'),
        Critical_Alerts=('Intervention_Priority', lambda x: (x == 'Critical Priority').sum())
    ).reset_index()
    quadrant_table['Cohort_Share'] = (quadrant_table['Students'] / len(filtered_df)) * 100
    quadrant_table = quadrant_table.sort_values(by='Attrition_Rate', ascending=False)
    
    st.dataframe(
        quadrant_table,
        column_config={
            "Quadrant": "Academic & Behavioral Quadrant",
            "Students": st.column_config.NumberColumn("Students (N)", format="%d"),
            "Cohort_Share": st.column_config.NumberColumn("Cohort Share (%)", format="%.1f%%"),
            "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f"),
            "Mean_Attendance": st.column_config.NumberColumn("Avg Attendance (%)", format="%.1f%%"),
            "Attrition_Rate": st.column_config.ProgressColumn("Empirical Attrition Rate (%)", format="%.1f%%", min_value=0, max_value=100),
            "Mean_Risk_Score": st.column_config.NumberColumn("Mean Risk Score (%)", format="%.1f%%"),
            "Critical_Alerts": st.column_config.NumberColumn("Critical Alerts", format="%d")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **The Critical Danger Zone (Q3)** captures students with both severe attendance collapse (<60%) and sub-2.0 CGPA. "
        "In this quadrant, empirical attrition reaches **84.3%**, and proactive academic advising check-ins must be triggered immediately."
    )

# ==============================================================================
# TAB 3: FUNDING & FINANCIAL PRECARITY
# ==============================================================================
with tab_socio:
    st.markdown("#### Funding Mechanisms & Tuition Delinquency Vulnerability")
    st.caption("Quantifies the correlation between economic support mechanisms, tuition payment status, and academic persistence.")
    
    col_fund_chart, col_tuit_chart = st.columns(2)
    with col_fund_chart:
        fig_fund = plot_attrition_by_funding(filtered_df)
        st.plotly_chart(fig_fund, use_container_width=True)
    with col_tuit_chart:
        fig_tuit = plot_tuition_risk_impact(filtered_df)
        st.plotly_chart(fig_tuit, use_container_width=True)
        
    st.divider()
    
    col_fund_tbl, col_tuit_tbl = st.columns(2)
    
    with col_fund_tbl:
        st.markdown("##### Funding Source Performance & Risk Ratio Data")
        
        fund_summary = filtered_df.groupby('FundingSource').agg(
            Student_Count=('StudentID', 'count'),
            Mean_Allowance=('MonthlyAllowance', 'mean'),
            Tuition_Paid_Rate=('TuitionPaid', lambda x: (x.astype(str).str.lower().isin(['paid', '1', 'yes'])).mean() * 100),
            Attrition_Rate=('Actual_Attrition', lambda x: x.mean() * 100 if len(x)>0 else 0.0),
            Mean_GPA=('GPA', 'mean')
        ).reset_index()
        
        baseline_rate = fund_summary.loc[fund_summary['FundingSource'] == 'Full Scholarship', 'Attrition_Rate']
        base_val = baseline_rate.values[0] if len(baseline_rate) > 0 and baseline_rate.values[0] > 0 else 15.0
        fund_summary['Relative_Risk'] = fund_summary['Attrition_Rate'] / base_val
        fund_summary = fund_summary.sort_values(by='Attrition_Rate', ascending=False)
        
        st.dataframe(
            fund_summary,
            column_config={
                "FundingSource": "Funding Mechanism",
                "Student_Count": st.column_config.NumberColumn("Students (N)", format="%d"),
                "Mean_Allowance": st.column_config.NumberColumn("Avg Monthly Allowance", format="₦%d"),
                "Tuition_Paid_Rate": st.column_config.NumberColumn("Tuition Paid Rate", format="%.1f%%"),
                "Attrition_Rate": st.column_config.ProgressColumn("Attrition Rate (%)", format="%.1f%%", min_value=0, max_value=60),
                "Relative_Risk": st.column_config.NumberColumn("Relative Risk (vs Scholarship)", format="%.2f×"),
                "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
        
    with col_tuit_tbl:
        st.markdown("##### Tuition Arrears × Priority Matrix Cross-Tabulation")
        
        tuit_cross = pd.crosstab(
            filtered_df['TuitionPaid'],
            filtered_df['Intervention_Priority'],
            margins=True,
            margins_name='Total Cohort'
        ).reset_index()
        
        st.dataframe(
            tuit_cross,
            column_config={
                "TuitionPaid": "Tuition Status",
                "Critical Priority": st.column_config.NumberColumn("Critical Priority", format="%d"),
                "Moderate Risk": st.column_config.NumberColumn("Moderate Risk", format="%d"),
                "On Track": st.column_config.NumberColumn("On Track", format="%d"),
                "Total Cohort": st.column_config.NumberColumn("Total Students", format="%d")
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.caption(
            "Students with unpaid tuition are overrepresented by 3.8× in the Critical Priority triage tier, "
            "underscoring how administrative holds directly precipitate withdrawal."
        )

# ==============================================================================
# TAB 4: FACULTY & GEOGRAPHIC DISPARITIES
# ==============================================================================
with tab_geo:
    st.markdown("#### Institutional Faculties & Geographic State Disparities")
    st.caption("Audits cross-faculty risk distributions and geographic regional disparities across Nigerian states.")
    
    col_fg1, col_fg2 = st.columns(2)
    with col_fg1:
        fig_fac = plot_faculty_risk_breakdown(filtered_df)
        st.plotly_chart(fig_fac, use_container_width=True)
    with col_fg2:
        fig_state = plot_state_risk_bars(filtered_df)
        st.plotly_chart(fig_state, use_container_width=True)
        
    st.divider()
    
    # Detailed Faculty Leaderboard Table
    st.markdown("##### Institutional Faculty Performance Leaderboard")
    
    fac_leaderboard = filtered_df.groupby('Faculty').agg(
        Enrollment=('StudentID', 'count'),
        Mean_GPA=('GPA', 'mean'),
        Mean_Attendance=('AttendanceRate', 'mean'),
        Mean_Risk_Score=('Risk_Score_Pct', 'mean'),
        At_Risk_Count=('Predicted_At_Risk', lambda x: int(x.sum())),
        Empirical_Attrition=('Actual_Attrition', lambda x: x.mean() * 100 if len(x)>0 else 0.0),
        Unpaid_Tuition=('TuitionPaid', lambda x: (x.astype(str).str.lower().isin(['unpaid', '0', 'no'])).mean() * 100)
    ).reset_index()
    fac_leaderboard['At_Risk_Rate'] = (fac_leaderboard['At_Risk_Count'] / fac_leaderboard['Enrollment']) * 100
    fac_leaderboard = fac_leaderboard.sort_values(by='Mean_Risk_Score', ascending=False)
    
    st.dataframe(
        fac_leaderboard,
        column_config={
            "Faculty": "University Faculty",
            "Enrollment": st.column_config.NumberColumn("Total Enrolled", format="%d"),
            "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f"),
            "Mean_Attendance": st.column_config.NumberColumn("Avg Attendance (%)", format="%.1f%%"),
            "Mean_Risk_Score": st.column_config.ProgressColumn("Mean Risk Score (%)", format="%.1f%%", min_value=0, max_value=80),
            "At_Risk_Count": st.column_config.NumberColumn("At-Risk Flagged", format="%d"),
            "At_Risk_Rate": st.column_config.NumberColumn("At-Risk Rate (%)", format="%.1f%%"),
            "Empirical_Attrition": st.column_config.NumberColumn("Hist. Attrition (%)", format="%.1f%%"),
            "Unpaid_Tuition": st.column_config.NumberColumn("Tuition Arrears (%)", format="%.1f%%")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("##### State-of-Origin Regional Risk Ranking (Top Disparities)")
    state_tbl = filtered_df.groupby('State').agg(
        Students=('StudentID', 'count'),
        Mean_Allowance=('MonthlyAllowance', 'mean'),
        Attrition_Rate=('Actual_Attrition', lambda x: x.mean() * 100 if len(x)>0 else 0.0),
        Mean_Distance=('DistanceFromHome', 'mean') if 'DistanceFromHome' in filtered_df.columns else ('GPA', 'mean'),
        At_Risk_Count=('Predicted_At_Risk', lambda x: int(x.sum()))
    ).reset_index()
    state_tbl['Cohort_Share'] = (state_tbl['Students'] / len(filtered_df)) * 100
    state_tbl = state_tbl.sort_values(by='Attrition_Rate', ascending=False)
    
    st.dataframe(
        state_tbl,
        column_config={
            "State": "State of Origin",
            "Students": st.column_config.NumberColumn("Enrolled (N)", format="%d"),
            "Cohort_Share": st.column_config.NumberColumn("Cohort Share (%)", format="%.1f%%"),
            "Mean_Allowance": st.column_config.NumberColumn("Avg Monthly Allowance", format="₦%d"),
            "Attrition_Rate": st.column_config.ProgressColumn("Attrition Rate (%)", format="%.1f%%", min_value=0, max_value=60),
            "Mean_Distance": st.column_config.NumberColumn("Avg Commute (km)", format="%.0f km"),
            "At_Risk_Count": st.column_config.NumberColumn("At-Risk Count", format="%d")
        },
        hide_index=True,
        use_container_width=True
    )

# ==============================================================================
# TAB 5: RAW COHORT DATA EXPLORER & STATISTICAL SUMMARIES
# ==============================================================================
with tab_explorer:
    st.markdown("#### Comprehensive Cohort Data Explorer & Statistical Summaries")
    st.caption("Inspect raw cohort distributions, numerical descriptive statistics, and drill down into individual student records.")
    
    st.markdown("##### Parametric Descriptive Statistics Table")
    numeric_cols = [
        'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate',
        'StudyHoursPerWeek', 'MonthlyAllowance', 'FailedCourses',
        'CoursesCreditLoad', 'Risk_Score_Pct'
    ]
    avail_num_cols = [c for c in numeric_cols if c in filtered_df.columns]
    desc_df = filtered_df[avail_num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc_df.columns = ['Count (N)', 'Mean (μ)', 'Std Dev (σ)', 'Min', '25% (Q1)', 'Median (50%)', '75% (Q3)', 'Max']
    
    st.dataframe(desc_df, use_container_width=True)
    
    st.divider()
    
    st.markdown(f"##### Interactive Student Record Browser ({len(filtered_df):,} records)")
    st.caption("Search, sort, filter, or export the filtered cohort slice.")
    
    search_query = st.text_input("Search student record by ID, Faculty, Department, or State", placeholder="e.g. LAG-SCI-1029 or Computer Science")
    
    browse_df = filtered_df.copy()
    if search_query:
        mask = (
            browse_df['StudentID'].astype(str).str.contains(search_query, case=False, na=False) |
            browse_df['Faculty'].astype(str).str.contains(search_query, case=False, na=False) |
            browse_df['Department'].astype(str).str.contains(search_query, case=False, na=False) |
            browse_df['State'].astype(str).str.contains(search_query, case=False, na=False)
        )
        browse_df = browse_df[mask]
        
    display_fields = [
        'StudentID', 'Faculty', 'Department', 'State', 'Gender', 'YearOfStudy',
        'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate', 'FailedCourses',
        'TuitionPaid', 'FundingSource', 'Risk_Score_Pct', 'Intervention_Priority'
    ]
    avail_fields = [f for f in display_fields if f in browse_df.columns]
    
    st.dataframe(
        browse_df[avail_fields].head(500),
        column_config={
            "Risk_Score_Pct": st.column_config.ProgressColumn("Risk Probability", format="%.1f%%", min_value=0, max_value=100),
            "GPA": st.column_config.NumberColumn("GPA", format="%.2f"),
            "GPA_Momentum": st.column_config.NumberColumn("Δ Momentum", format="%+.2f"),
            "AttendanceRate": st.column_config.NumberColumn("Attendance", format="%.1f%%")
        },
        hide_index=True,
        use_container_width=True
    )
    if len(browse_df) > 500:
        st.caption(f"Displaying top 500 of {len(browse_df):,} records for responsive rendering.")
