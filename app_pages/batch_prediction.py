import sys
import os
from pathlib import Path
import importlib
import streamlit as st
import pandas as pd
import numpy as np

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import fedra_app.pipeline
importlib.reload(fedra_app.pipeline)

from fedra_app.pipeline import (
    train_suite_models,
    predict_batch,
    get_csv_template,
    get_sample_dataframe,
    FACULTIES,
    STATES
)
from fedra_app.visualizations import (
    plot_batch_risk_distribution,
    plot_batch_priority_donut,
    plot_batch_faculty_bars
)

st.header("Batch student risk prediction & CSV upload", anchor=False)
st.caption("Upload institutional rosters or class cohorts to run FEDRA 3.0 multi-scale ensemble attrition predictions, triage risk tiers, and generate actionable intervention recommendations.")

with st.spinner("Initializing FEDRA 3.0 ensemble pipeline..."):
    artifacts = train_suite_models()
    th_fedra = artifacts['eval']['FEDRA 3.0']['threshold']

# ==============================================================================
# 1. TEMPLATE DOWNLOAD & SCHEMA GUIDANCE
# ==============================================================================
with st.container(border=True):
    st.subheader("📥 CSV Templates & Format Specification", anchor=False)
    st.markdown(
        "Download a pre-formatted template to ensure your institutional dataset matches the FEDRA 3.0 schema. "
        "You can download either a **blank header template** or a **sample populated batch** containing 10 diverse Nigerian student profiles."
    )
    
    col_dl1, col_dl2, col_demo = st.columns([1, 1, 1.2])
    
    with col_dl1:
        blank_csv = get_csv_template(sample=False)
        st.download_button(
            label="📄 Download blank template",
            data=blank_csv,
            file_name="fedra3_student_template_blank.csv",
            mime="text/csv",
            help="CSV containing the canonical 26 column headers ready for data entry.",
            use_container_width=True
        )
        
    with col_dl2:
        sample_csv = get_csv_template(sample=True)
        st.download_button(
            label="📊 Download sample data CSV",
            data=sample_csv,
            file_name="fedra3_sample_students_10.csv",
            mime="text/csv",
            help="CSV with 10 representative Nigerian student profiles for immediate testing.",
            use_container_width=True
        )
        
    with col_demo:
        if st.button("⚡ Load demo sample data (10 students)", use_container_width=True, type="secondary"):
            st.session_state['batch_df_raw'] = get_sample_dataframe()
            st.session_state['batch_source_name'] = "Demo Sample (10 students)"
            if 'batch_predictions' in st.session_state:
                del st.session_state['batch_predictions']
            st.rerun()

    with st.expander("📋 View Expected CSV Column Schema & Guidelines", expanded=False):
        c_sch1, c_sch2 = st.columns(2)
        with c_sch1:
            st.markdown("""
            **Academic & Behavioral Indicators:**
            - `StudentID`: Unique student identifier (e.g. `STU-1001`)
            - `GPA`: Current estimated GPA on a 5.0 scale (e.g. `3.45`)
            - `PreviousSemesterGPA`: Prior term GPA on 5.0 scale (e.g. `3.20`)
            - `AttendanceRate`: Class attendance percentage `0`–`100` (e.g. `85.0` or `85%`)
            - `StudyHoursPerWeek`: Weekly independent study hours (e.g. `15`)
            - `FailedCourses`: Number of registered courses failed (e.g. `0`, `2`)
            - `CoursesCreditLoad`: Total credit unit load (e.g. `18`)
            - `MentoringSessionsAttended`: Peer/faculty mentoring sessions (e.g. `2`)
            - `CounsellingVisits`: Institutional guidance visits (e.g. `0`)
            - `LibraryVisitsPerMonth`: Monthly physical/digital library visits (e.g. `4`)
            """)
        with c_sch2:
            st.markdown("""
            **Socioeconomic, Demographic & Institutional Context:**
            - `TuitionPaid`: Tuition fee status (`Paid` or `Unpaid`, `1` or `0`)
            - `MonthlyAllowance`: Monthly living allowance in Naira (e.g. `45000` or `45k`)
            - `FundingSource`: `Self Funded`, `Government Loan`, `Partial Scholarship`, `Full Scholarship`, `Family Support`
            - `Housing`: `Off Campus`, `On Campus`, `Family Home`
            - `DistanceFromHome`: Commute distance from residence in km (e.g. `35`)
            - `Faculty`: Target faculty (e.g. `Faculty of Science`, `Faculty of Engineering`)
            - `Department`: Student major/department (e.g. `Computer Science`)
            - `State`: State of origin (e.g. `Lagos`, `Kano`, `Plateau`)
            - `Gender`: `Male`, `Female`
            - `ParentalEducationLevel`: `Primary`, `Secondary`, `Tertiary`
            """)
        st.caption("ℹ️ *Note: Missing or unrecognized columns will automatically fall back to balanced median/mode baseline values so prediction never fails.*")

st.divider()

# ==============================================================================
# 2. FILE UPLOADER & DATA PREPARATION
# ==============================================================================
st.subheader("📤 Upload Student CSV for Prediction", anchor=False)

uploaded_file = st.file_uploader(
    "Choose a CSV file containing student records",
    type=["csv"],
    help="Upload your formatted CSV file to initiate batch scoring."
)

# Handle file upload
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if 'Unnamed: 0' in raw_df.columns:
            raw_df.drop(columns=['Unnamed: 0'], inplace=True)
            
        st.session_state['batch_df_raw'] = raw_df
        st.session_state['batch_source_name'] = uploaded_file.name
        # Clear previous predictions if file changed
        if st.session_state.get('last_uploaded_name') != uploaded_file.name:
            st.session_state['last_uploaded_name'] = uploaded_file.name
            if 'batch_predictions' in st.session_state:
                del st.session_state['batch_predictions']
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")

# Check if data is loaded in session
if 'batch_df_raw' in st.session_state:
    df_current = st.session_state['batch_df_raw']
    src_name = st.session_state.get('batch_source_name', 'Uploaded File')
    
    st.info(f"📁 Loaded dataset: **{src_name}** | **{len(df_current):,}** student records detected across **{len(df_current.columns)}** columns.")
    
    with st.expander("👁️ Preview Raw Uploaded Data (First 5 Rows)", expanded=False):
        st.dataframe(df_current.head(5), width='stretch')

    # Prediction Action Button
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_pred = st.button("🔮 Run FEDRA 3.0 Batch Prediction", type="primary", use_container_width=True)
        
    with col_info:
        st.caption(f"Ensemble Models: CatBoost (w=0.45), XGBoost (w=0.45), Logistic Anchor (w=0.10) | Calibrated Cutoff: **{th_fedra*100:.1f}%**")
        
    if run_pred or ('batch_predictions' in st.session_state):
        if run_pred or ('batch_predictions' not in st.session_state):
            with st.spinner("Transforming features and computing multi-branch predictions..."):
                try:
                    pred_results = predict_batch(df_current, artifacts=artifacts)
                    st.session_state['batch_predictions'] = pred_results
                except Exception as ex:
                    st.error(f"Prediction failed: {ex}")
                    st.stop()
                    
        res_df = st.session_state['batch_predictions']
        
        st.divider()
        
        # ==============================================================================
        # 3. KPI SUMMARY CARDS
        # ==============================================================================
        total_n = len(res_df)
        at_risk_n = int(res_df['Predicted_At_Risk'].sum())
        at_risk_pct = (at_risk_n / total_n * 100) if total_n > 0 else 0.0
        critical_n = int((res_df['Intervention_Priority'] == 'Critical Priority').sum())
        critical_pct = (critical_n / total_n * 100) if total_n > 0 else 0.0
        avg_risk = float(res_df['Risk_Score_Pct'].mean()) if total_n > 0 else 0.0
        
        st.subheader("📊 Batch Assessment Summary & Early-Warning KPIs", anchor=False)
        
        with st.container(horizontal=True):
            st.metric("Total Records Evaluated", f"{total_n:,}", "Students in batch", border=True)
            st.metric(
                "Flagged At-Risk",
                f"{at_risk_n:,}",
                f"{at_risk_pct:.1f}% of batch",
                delta_color="inverse",
                border=True
            )
            st.metric(
                "Critical Priority Alerts",
                f"{critical_n:,}",
                f"{critical_pct:.1f}% of batch",
                delta_color="inverse",
                border=True
            )
            st.metric(
                "Batch Mean Risk Score",
                f"{avg_risk:.1f}%",
                f"Cutoff: {th_fedra*100:.1f}%",
                delta_color="inverse",
                border=True
            )

        # ==============================================================================
        # 4. VISUAL DISTRIBUTION & BREAKDOWN CHARTS
        # ==============================================================================
        st.markdown("##### Batch Risk Visualizations & Distributions")
        col_c1, col_c2 = st.columns([1.3, 1])
        
        with col_c1:
            fig_dist = plot_batch_risk_distribution(res_df, threshold=th_fedra)
            st.plotly_chart(fig_dist, width='stretch')
            
        with col_c2:
            fig_donut = plot_batch_priority_donut(res_df)
            st.plotly_chart(fig_donut, width='stretch')

        if 'Faculty' in res_df.columns and res_df['Faculty'].nunique() > 1:
            fig_fac = plot_batch_faculty_bars(res_df)
            if fig_fac:
                st.plotly_chart(fig_fac, width='stretch')

        st.divider()

        # ==============================================================================
        # 5. FILTERABLE STUDENT INTERVENTION QUEUE
        # ==============================================================================
        st.subheader(f"📋 Scored Student Intervention Queue ({total_n:,} students)", anchor=False)
        
        # Filter row
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
        with f_col1:
            prio_filter = st.multiselect(
                "Filter by Triage Priority",
                ["Critical Priority", "Moderate Risk", "On Track"],
                default=["Critical Priority", "Moderate Risk", "On Track"]
            )
        with f_col2:
            fac_options = ["All Faculties"] + sorted([str(x) for x in res_df['Faculty'].dropna().unique()]) if 'Faculty' in res_df.columns else ["All Faculties"]
            fac_filter = st.selectbox("Filter by Faculty", fac_options)
            
        with f_col3:
            search_query = st.text_input("🔍 Search StudentID, Department, State...", placeholder="Type to filter...")

        # Apply filters
        filtered_res = res_df.copy()
        if prio_filter:
            filtered_res = filtered_res[filtered_res['Intervention_Priority'].isin(prio_filter)]
        if fac_filter != "All Faculties" and 'Faculty' in filtered_res.columns:
            filtered_res = filtered_res[filtered_res['Faculty'] == fac_filter]
        if search_query:
            sq = search_query.strip().lower()
            mask = filtered_res.astype(str).apply(lambda row: sq in ' '.join(row).lower(), axis=1)
            filtered_res = filtered_res[mask]

        st.caption(f"Showing **{len(filtered_res):,}** of **{total_n:,}** student records.")

        display_queue_cols = [
            'StudentID', 'Faculty', 'Department', 'State', 'Gender',
            'GPA', 'PreviousSemesterGPA', 'GPA_Momentum', 'AttendanceRate', 'TuitionPaid',
            'Risk_Score_Pct', 'Intervention_Priority', 'Actionable_Recommendation'
        ]
        # Only select columns that exist in DataFrame
        queue_cols = [c for c in display_queue_cols if c in filtered_res.columns]

        st.dataframe(
            filtered_res[queue_cols],
            column_config={
                "Risk_Score_Pct": st.column_config.ProgressColumn(
                    "Dropout Risk",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "GPA": st.column_config.NumberColumn("Current GPA", format="%.2f"),
                "GPA_Momentum": st.column_config.NumberColumn("Δ Momentum", format="%+.2f"),
                "AttendanceRate": st.column_config.NumberColumn("Attendance", format="%.1f%%"),
                "Intervention_Priority": st.column_config.TextColumn("Triage Tier"),
                "Actionable_Recommendation": st.column_config.TextColumn("Personalized Intervention Action")
            },
            width='stretch',
            hide_index=True
        )

        st.divider()

        # ==============================================================================
        # 6. DOWNLOAD SCORED PREDICTIONS
        # ==============================================================================
        st.subheader("💾 Export Scored Predictions", anchor=False)
        st.markdown("Export the enriched predictions with risk scores, probabilities, and actionable recommendations as CSV.")

        c_exp1, c_exp2 = st.columns(2)
        
        with c_exp1:
            full_csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Scored Batch (CSV)",
                data=full_csv,
                file_name="fedra3_batch_predictions_full.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            
        with c_exp2:
            at_risk_df = res_df[res_df['Predicted_At_Risk'] == 1]
            at_risk_csv = at_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"🚨 Download Flagged At-Risk Only ({len(at_risk_df):,} Students)",
                data=at_risk_csv,
                file_name="fedra3_batch_predictions_at_risk_only.csv",
                mime="text/csv",
                disabled=(len(at_risk_df) == 0),
                use_container_width=True
            )
else:
    st.info("👆 Please upload a student CSV file above, or click **⚡ Load demo sample data** to preview the batch prediction workflow.")

