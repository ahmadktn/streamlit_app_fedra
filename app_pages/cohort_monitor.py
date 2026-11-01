import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fedra_app.pipeline import train_suite_models, FACULTIES, STATES
from fedra_app.visualizations import plot_cohort_risk_histogram, plot_faculty_risk_breakdown

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
col_filter, col_chart = st.columns([1, 2])

with col_filter:
    with st.container(border=True):
        st.subheader("Cohort filters", anchor=False)
        selected_priority = st.multiselect(
            "Intervention priority",
            ["Critical Priority", "Moderate Risk", "On Track"],
            default=["Critical Priority", "Moderate Risk"]
        )
        selected_faculty = st.selectbox("Faculty filter", ["All Faculties"] + FACULTIES)
        selected_tuition = st.selectbox("Tuition payment status", ["All", "Paid", "Unpaid"])
        selected_funding = st.selectbox("Funding source", ["All", "Self Funded", "Government Loan", "Partial Scholarship", "Full Scholarship", "Family Support"])

# Filter logic
filtered_df = df.copy()
if selected_priority:
    filtered_df = filtered_df[filtered_df['Intervention_Priority'].isin(selected_priority)]
if selected_faculty != "All Faculties":
    filtered_df = filtered_df[filtered_df['Faculty'] == selected_faculty]
if selected_tuition != "All":
    filtered_df = filtered_df[filtered_df['TuitionPaid'] == selected_tuition]
if selected_funding != "All":
    filtered_df = filtered_df[filtered_df['FundingSource'] == selected_funding]

with col_chart:
    with st.container(border=True):
        tab_dist, tab_fac = st.tabs(["Cohort risk distribution", "Faculty risk ranking"])
        with tab_dist:
            fig_hist = plot_cohort_risk_histogram(filtered_df if len(filtered_df) > 0 else df, threshold=th_fedra)
            st.plotly_chart(fig_hist, use_container_width=True)
        with tab_fac:
            fig_fac = plot_faculty_risk_breakdown(filtered_df if len(filtered_df) > 0 else df)
            st.plotly_chart(fig_fac, use_container_width=True)

# 3. Triage Queue Table
with st.container(border=True):
    st.subheader(f"Filtered student intervention queue ({len(filtered_df):,} students)", anchor=False)
    
    display_cols = [
        'StudentID', 'Faculty', 'Department', 'State', 'Gender',
        'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate', 'TuitionPaid',
        'Risk_Score_Pct', 'Intervention_Priority', 'Actionable_Recommendation'
    ]
    
    st.dataframe(
        filtered_df[display_cols].head(500),
        column_config={
            "Risk_Score_Pct": st.column_config.ProgressColumn(
                "Risk probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "GPA": st.column_config.NumberColumn("Current GPA", format="%.2f"),
            "GPA_Momentum": st.column_config.NumberColumn("Δ GPA Velocity", format="%+.2f"),
            "AttendanceRate": st.column_config.NumberColumn("Attendance", format="%.1f%%")
        },
        use_container_width=True,
        hide_index=True
    )
    if len(filtered_df) > 500:
        st.caption(f"Displaying top 500 of {len(filtered_df):,} matching student records for responsive rendering.")
