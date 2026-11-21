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
    ALL_DEPARTMENTS
)
from fedra_app.visualizations import (
    plot_cohort_risk_histogram,
    plot_faculty_risk_breakdown,
    plot_department_risk_bars
)

st.header("Cohort early warning monitor (Nigerian Higher-Ed)", anchor=False)
st.caption("Institutional failure & dropout risk triage powered by FEDRA 3.0 multi-scale trajectory dynamics (N=50,150).")

with st.spinner("Loading Nigerian student cohort and triage distributions..."):
    artifacts = train_suite_models()

df = artifacts['data']['df'].copy()
th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']

# 1. KPI Row
total_students = len(df)
at_risk_count = int(df['Predicted_At_Risk'].sum())
at_risk_pct = at_risk_count / total_students * 100
critical_count = int((df['Intervention_Priority'] == 'Critical Priority').sum())
avg_gpa = float(df['GPA'].mean())

with st.container(horizontal=True):
    st.metric("Total enrolled cohort", f"{total_students:,}", "Active university students", border=True)
    st.metric("At-risk students flagged", f"{at_risk_count:,}", f"{at_risk_pct:.1f}% cohort", delta_color="inverse", border=True)
    st.metric("Critical priority alerts", f"{critical_count:,}", "Risk score >= 60%", delta_color="inverse", border=True)
    st.metric("Cohort average GPA", f"{avg_gpa:.2f} / 5.0", "5.0 CGPA Scale", border=True)

st.divider()

# 2. Filters & Visual Distribution
col_filter, col_chart = st.columns([1.1, 1.9])

with col_filter:
    with st.container(border=True):
        st.subheader("Cohort filters", anchor=False)
        selected_priority = st.multiselect(
            "Intervention priority",
            ["Critical Priority", "Moderate Risk", "On Track"],
            default=["Critical Priority", "Moderate Risk"]
        )
        selected_faculty = st.selectbox("Faculty filter", ["All Faculties"] + FACULTIES)
        
        # Cascading Department filter for departmental isolation
        if selected_faculty != "All Faculties":
            available_depts = FACULTY_DEPARTMENTS_MAP.get(selected_faculty, [])
        else:
            available_depts = ALL_DEPARTMENTS
        selected_dept = st.selectbox("Department (One at a time)", ["All Departments"] + available_depts)
        
        # Year of Study filter
        year_options = ["All Years", "Year 1 (100L)", "Year 2 (200L)", "Year 3 (300L)", "Year 4 (400L)", "Year 5 (500L)"]
        selected_year_str = st.selectbox("Year of study (Level)", year_options, index=0)
        
        selected_tuition = st.selectbox("Tuition payment status", ["All", "Paid", "Unpaid"])
        selected_funding = st.selectbox("Funding source", ["All", "Self Funded", "Government Loan", "Partial Scholarship", "Full Scholarship", "Family Support"])

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
if selected_tuition != "All":
    filtered_df = filtered_df[filtered_df['TuitionPaid'] == selected_tuition]
if selected_funding != "All":
    filtered_df = filtered_df[filtered_df['FundingSource'] == selected_funding]

with col_chart:
    with st.container(border=True):
        tab_dist, tab_fac, tab_dept = st.tabs(["Cohort risk distribution", "Faculty risk ranking", "Department risk ranking"])
        active_cohort = filtered_df if len(filtered_df) > 0 else df
        
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
