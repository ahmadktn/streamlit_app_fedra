import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fedra_app.pipeline import (
    train_suite_models,
    FACULTIES,
    STATES,
    FACULTY_DEPARTMENTS_MAP,
    ALL_DEPARTMENTS,
    get_faculty_for_department
)
from fedra_app.visualizations import (
    plot_cohort_risk_histogram,
    plot_faculty_risk_breakdown,
    plot_department_risk_bars
)

# Synchronize with global sidebar department scope
global_dept = st.session_state.get("active_department", "All Departments (Overview)")

if global_dept != "All Departments (Overview)":
    parent_fac = get_faculty_for_department(global_dept)
    st.header(f"Cohort Early Warning Monitor: {global_dept}", anchor=False)
    st.caption(f"Isolated departmental risk triage for {global_dept} ({parent_fac}). Early-warning dynamics calibrated to program-specific baselines.")
else:
    st.header("Cohort Early Warning Monitor (University-Wide)", anchor=False)
    st.caption("Institutional failure and dropout risk triage across all faculties (N=50,150). Select a department in the sidebar to isolate program risks.")

with st.spinner("Loading student cohort and triage distributions..."):
    artifacts = train_suite_models()

df = artifacts['data']['df'].copy()
th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']

# Initial department isolation from global scope
if global_dept != "All Departments (Overview)" and 'Department' in df.columns:
    base_df = df[df['Department'] == global_dept].copy()
else:
    base_df = df.copy()

# ==============================================================================
# 1. FILTERS & SUBGROUP SELECTION
# ==============================================================================
with st.expander("Filter Cohort Records", expanded=True):
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    with f_c1:
        default_fac_idx = 0
        if global_dept != "All Departments (Overview)":
            p_fac = get_faculty_for_department(global_dept)
            if p_fac in FACULTIES:
                default_fac_idx = FACULTIES.index(p_fac) + 1
        selected_faculty = st.selectbox(
            "Faculty Filter",
            ["All Faculties"] + FACULTIES,
            index=default_fac_idx
        )
    with f_c2:
        if selected_faculty != "All Faculties":
            available_depts = FACULTY_DEPARTMENTS_MAP.get(selected_faculty, ALL_DEPARTMENTS)
        else:
            available_depts = ALL_DEPARTMENTS
            
        dept_options_local = ["All Departments"] + available_depts
        default_dept_idx = 0
        if global_dept != "All Departments (Overview)" and global_dept in dept_options_local:
            default_dept_idx = dept_options_local.index(global_dept)
            
        selected_dept = st.selectbox(
            "Department Filter",
            dept_options_local,
            index=default_dept_idx
        )
    with f_c3:
        year_options = ["All Years", "Year 1 (100L)", "Year 2 (200L)", "Year 3 (300L)", "Year 4 (400L)", "Year 5 (500L)"]
        selected_year_str = st.selectbox("Year of Study (Level)", year_options, index=0)
    with f_c4:
        selected_priority = st.multiselect(
            "Intervention Priority",
            ["Critical Priority", "Moderate Risk", "On Track"],
            default=["Critical Priority", "Moderate Risk", "On Track"]
        )

# Filter logic
filtered_df = df.copy()
if selected_priority:
    filtered_df = filtered_df[filtered_df['Intervention_Priority'].isin(selected_priority)]
if selected_faculty != "All Faculties":
    filtered_df = filtered_df[filtered_df['Faculty'] == selected_faculty]
if selected_dept != "All Departments" and 'Department' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
if selected_year_str != "All Years" and 'YearOfStudy' in filtered_df.columns:
    year_val = selected_year_str.split()[1] # extracts '1', '2', etc.
    filtered_df = filtered_df[filtered_df['YearOfStudy'].astype(str) == year_val]

# ==============================================================================
# 2. DYNAMIC KPI ROW (COMPUTED ON FILTERED DEPARTMENT COHORT)
# ==============================================================================
total_students = len(filtered_df)
at_risk_count = int(filtered_df['Predicted_At_Risk'].sum()) if total_students > 0 else 0
at_risk_pct = (at_risk_count / total_students * 100) if total_students > 0 else 0.0
critical_count = int((filtered_df['Intervention_Priority'] == 'Critical Priority').sum()) if total_students > 0 else 0
critical_pct = (critical_count / total_students * 100) if total_students > 0 else 0.0
avg_gpa = float(filtered_df['GPA'].mean()) if total_students > 0 else 0.0

scope_label = selected_dept if selected_dept != "All Departments" else (global_dept if global_dept != "All Departments (Overview)" else "All Enrolled Students")

st.markdown(f"##### Key Risk Metrics: {scope_label}")
with st.container(horizontal=True):
    st.metric("Total Enrolled Cohort", f"{total_students:,}", f"Scope: {scope_label}", border=True)
    st.metric("At-Risk Students Flagged", f"{at_risk_count:,}", f"{at_risk_pct:.1f}% of cohort", delta_color="inverse", border=True)
    st.metric("Critical Priority Alerts", f"{critical_count:,}", f"{critical_pct:.1f}% critical", delta_color="inverse", border=True)
    st.metric("Cohort Average GPA", f"{avg_gpa:.2f} / 5.0", "5.0 CGPA Scale", border=True)

st.divider()

# ==============================================================================
# 3. VISUAL DISTRIBUTION & BREAKDOWN CHARTS
# ==============================================================================
active_cohort = filtered_df if len(filtered_df) > 0 else df
tab_dist, tab_fac, tab_dept = st.tabs(["Cohort risk distribution", "Faculty risk ranking", "Department risk ranking"])

with tab_dist:
    fig_hist = plot_cohort_risk_histogram(active_cohort, threshold=th_fedra)
    st.plotly_chart(fig_hist, width='stretch')
    
    st.markdown("**Risk Tier Breakdown Table**")
    tier_summary = active_cohort.groupby('Intervention_Priority').agg(
        Students=('StudentID', 'count'),
        Mean_GPA=('GPA', 'mean'),
        Mean_Attendance=('AttendanceRate', 'mean'),
        Mean_Risk_Score=('Risk_Score_Pct', 'mean')
    ).reset_index()
    tier_summary['Cohort_Share'] = (tier_summary['Students'] / len(active_cohort)) * 100
    
    st.dataframe(
        tier_summary,
        column_config={
            "Intervention_Priority": "Triage Priority Tier",
            "Students": st.column_config.NumberColumn("Students (N)", format="%d"),
            "Cohort_Share": st.column_config.NumberColumn("Share (%)", format="%.1f%%"),
            "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f"),
            "Mean_Attendance": st.column_config.NumberColumn("Avg Attendance", format="%.1f%%"),
            "Mean_Risk_Score": st.column_config.NumberColumn("Avg Risk", format="%.1f%%")
        },
        hide_index=True,
        width='stretch'
    )
    
with tab_fac:
    fig_fac = plot_faculty_risk_breakdown(active_cohort)
    st.plotly_chart(fig_fac, width='stretch')
    
    st.markdown("**Faculty Risk Data Leaderboard**")
    fac_tbl = active_cohort.groupby('Faculty').agg(
        Students=('StudentID', 'count'),
        Mean_Risk=('Risk_Score_Pct', 'mean'),
        At_Risk_Students=('Predicted_At_Risk', lambda x: int(x.sum())),
        Mean_GPA=('GPA', 'mean')
    ).reset_index().sort_values(by='Mean_Risk', ascending=False)
    fac_tbl['At_Risk_Rate'] = (fac_tbl['At_Risk_Students'] / fac_tbl['Students']) * 100
    
    st.dataframe(
        fac_tbl,
        column_config={
            "Faculty": "Faculty",
            "Students": st.column_config.NumberColumn("Enrolled (N)", format="%d"),
            "Mean_Risk": st.column_config.ProgressColumn("Mean Risk (%)", format="%.1f%%", min_value=0, max_value=80),
            "At_Risk_Students": st.column_config.NumberColumn("Flagged At-Risk", format="%d"),
            "At_Risk_Rate": st.column_config.NumberColumn("At-Risk Rate (%)", format="%.1f%%"),
            "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f")
        },
        hide_index=True,
        width='stretch'
    )

with tab_dept:
    if 'Department' in active_cohort.columns and active_cohort['Department'].nunique() > 1:
        fig_dept = plot_department_risk_bars(active_cohort, selected_dept=selected_dept if selected_dept != "All Departments" else None, max_departments=12)
        if fig_dept:
            st.plotly_chart(fig_dept, width='stretch')
            
        st.markdown("**Department Risk Data Leaderboard**")
        dept_tbl = active_cohort.groupby('Department').agg(
            Students=('StudentID', 'count'),
            Mean_Risk=('Risk_Score_Pct', 'mean'),
            At_Risk_Students=('Predicted_At_Risk', lambda x: int(x.sum())),
            Mean_GPA=('GPA', 'mean')
        ).reset_index().sort_values(by='Mean_Risk', ascending=False)
        dept_tbl['At_Risk_Rate'] = (dept_tbl['At_Risk_Students'] / dept_tbl['Students']) * 100
        
        st.dataframe(
            dept_tbl.head(20),
            column_config={
                "Department": "Department",
                "Students": st.column_config.NumberColumn("Enrolled (N)", format="%d"),
                "Mean_Risk": st.column_config.ProgressColumn("Mean Risk (%)", format="%.1f%%", min_value=0, max_value=80),
                "At_Risk_Students": st.column_config.NumberColumn("Flagged At-Risk", format="%d"),
                "At_Risk_Rate": st.column_config.NumberColumn("At-Risk Rate (%)", format="%.1f%%"),
                "Mean_GPA": st.column_config.NumberColumn("Avg GPA", format="%.2f")
            },
            hide_index=True,
            width='stretch'
        )
    else:
        dept_name = selected_dept if selected_dept != "All Departments" else "the selected department"
        st.info(f"Currently focused on **{dept_name}** ({len(active_cohort):,} students). Select 'All Departments' to view cross-department comparisons.")


# 3. Triage Queue Table
with st.container(border=True):
    st.subheader(f"Filtered student intervention queue ({len(filtered_df):,} students)", anchor=False)
    
    display_cols = [
        'StudentID', 'Faculty', 'Department', 'YearOfStudy', 'State', 'Gender',
        'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate', 'TuitionPaid',
        'Risk_Score_Pct', 'Intervention_Priority', 'Actionable_Recommendation'
    ]
    # Filter columns that exist
    actual_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[actual_cols].head(500),
        column_config={
            "Risk_Score_Pct": st.column_config.ProgressColumn(
                "Risk probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "GPA": st.column_config.NumberColumn("Current GPA", format="%.2f"),
            "GPA_Momentum": st.column_config.NumberColumn("Δ GPA Velocity", format="%+.2f"),
            "AttendanceRate": st.column_config.NumberColumn("Attendance", format="%.1f%%"),
            "YearOfStudy": st.column_config.TextColumn("Year")
        },
        width='stretch',
        hide_index=True
    )
    if len(filtered_df) > 500:
        st.caption(f"Displaying top 500 of {len(filtered_df):,} matching student records for responsive rendering.")
